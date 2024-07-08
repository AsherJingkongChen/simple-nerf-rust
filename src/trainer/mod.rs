use crate::*;

use anyhow::{anyhow, Result};
use burn::{
    data::dataset::{transform, Dataset},
    module::AutodiffModule,
    nn::loss,
    optim::{self, Optimizer},
    prelude::*,
    record,
    tensor::backend::AutodiffBackend,
};
use kdam::{term, tqdm, Bar, BarExt};
use std::{
    fs,
    io::{stderr, IsTerminal},
    path::PathBuf,
};

#[derive(Config)]
pub struct TrainerConfig {
    pub artifact_directory: String,
    pub dataset: dataset::SimpleNerfDatasetConfig,
    pub dataset_file_path_or_url: String,
    pub learning_rate: f64,
    pub epoch_count: usize,
    pub train_ratio: f32,
    pub renderer: renderer::VolumeRendererConfig,
    pub seed: Option<u64>,
}

pub struct Trainer<B: AutodiffBackend> {
    artifact_directory: PathBuf,
    criterion: loss::MseLoss<B>,
    dataset_test: dataset::SimpleNerfDataset<B>,
    dataset_train: transform::SamplerDataset<
        dataset::SimpleNerfDataset<B>,
        dataset::SimpleNerfDatasetItem,
    >,
    device: B::Device,
    epoch_count: usize,
    metric: metric::PsnrMetric<B::InnerBackend>,
    learning_rate: f64,
    progress_bar: Bar,
    seed: Option<u64>,
    renderer: renderer::VolumeRenderer<B>,
}

impl TrainerConfig {
    pub fn init<B: AutodiffBackend>(
        &self,
        device: &B::Device,
    ) -> Result<Trainer<B>> {
        if let Some(seed) = self.seed {
            B::seed(seed);
        }

        let artifact_directory = PathBuf::from(&self.artifact_directory);

        let criterion = loss::MseLoss::new();

        let (dataset_test, dataset_train) = {
            let datasets = self
                .dataset
                .init_from_file_path_or_url(
                    &self.dataset_file_path_or_url,
                    device,
                )?
                .split_for_training(self.train_ratio);
            let dataset_train_size = datasets.train.len();
            (
                datasets.test,
                transform::SamplerDataset::new(
                    datasets.train,
                    dataset_train_size,
                ),
            )
        };

        let metric = metric::PsnrMetric::<B::InnerBackend>::init(device);

        let renderer = self.renderer.init(device)?;

        let progress_bar = {
            let mut bar = tqdm!(
                desc = format!("Training on {} items", dataset_train.len()),
                colour = "orangered",
                dynamic_ncols = true,
                force_refresh = true,
                total = self.epoch_count,
                unit = "steps",
                bar_format = "{desc suffix=''} {postfix} ┃ \
                {percentage:.0}% = {count}/{total} {unit} ┃ \
                {rate:.1} {unit}/s ┃ \
                {remaining human=true} \
                ┃{animation}┃"
            );
            bar.postfix = format!("┃ PSNR = 0.00 dB");
            bar
        };

        let _ = fs::remove_dir_all(&artifact_directory);
        fs::create_dir_all(&artifact_directory)?;
        self.save(artifact_directory.join("config.json"))?;

        Ok(Trainer {
            artifact_directory,
            criterion,
            dataset_train,
            dataset_test,
            device: device.clone(),
            epoch_count: self.epoch_count,
            learning_rate: self.learning_rate,
            metric,
            renderer,
            progress_bar,
            seed: self.seed,
        })
    }
}

impl<B: AutodiffBackend> Trainer<B> {
    pub fn fit(mut self) -> Result<()> {
        let mut optimizer = optim::AdamConfig::new().init();

        if let Some(seed) = self.seed {
            B::seed(seed);
        }
        term::init(stderr().is_terminal());
        self.progress_bar.reset(None);

        // Training Loop
        for epoch in 0..self.epoch_count {
            let input = self.dataset_train.get(0);
            if input.is_none() {
                break;
            }
            let input = input.unwrap().into_batch(&self.device);

            let output_image = self.renderer.forward(
                input.directions,
                input.intervals,
                input.positions,
            );
            eprintln!(
                "input_image.mean(): {:?}",
                input.image.clone().mean().into_scalar()
            );
            eprintln!(
                "output_image.mean(): {:?}",
                output_image.clone().mean().into_scalar()
            );

            let loss = self.criterion.forward(
                output_image,
                input.image,
                loss::Reduction::Mean,
            );
            eprintln!("loss: {:?}", loss.clone().into_scalar());

            let gradients = optim::GradientsParams::from_grads(
                loss.backward(),
                &self.renderer,
            );
            self.renderer =
                optimizer.step(self.learning_rate, self.renderer, gradients);

            // Monitoring
            if epoch % 20 == 0 && epoch > 0 {
                let input = self.dataset_test.get(0);
                if input.is_none() {
                    continue;
                }
                let input = input.unwrap().into_batch(&self.device);

                let output_image = self.renderer.valid().forward(
                    input.directions,
                    input.intervals,
                    input.positions,
                );

                let score_fidelity = self
                    .metric
                    .forward(output_image, input.image)
                    .into_scalar();

                self.progress_bar.postfix =
                    format!("┃ PSNR = {:.2} dB", score_fidelity);
            }

            self.progress_bar.update(1)?;
        }

        // End of Training Loop
        {
            self.progress_bar.clear()?;
            self.progress_bar
                .set_bar_format(
                    "{desc suffix=''} ┃ \
                    {total} {unit} ┃ \
                    {rate:.1} {unit}/s ┃ \
                    {elapsed human=true}\n",
                )
                .map_err(|e| anyhow!(e))?;
            self.progress_bar.set_description(format!(
                "Trained on {} items",
                self.dataset_train.len()
            ));
            self.progress_bar.refresh()?;
        }

        // Testing
        println!("Testing on {} items", self.dataset_test.len());
        let renderer = self.renderer.valid();
        for (index, input) in self.dataset_test.iter().enumerate() {
            let input = input.into_batch(&self.device);
            let output_image = renderer.forward(
                input.directions,
                input.intervals,
                input.positions,
            );
            let score_fidelity =
                self.metric.forward(output_image, input.image).into_scalar();
            println!("No. {:03} ┃ PSNR = {:.2} dB", index + 1, score_fidelity);
        }

        // Save the Renderer
        self.renderer.save_file(
            self.artifact_directory.join("renderer"),
            &record::NamedMpkFileRecorder::<record::FullPrecisionSettings>::new(
            ),
        )?;

        Ok(())
    }
}
