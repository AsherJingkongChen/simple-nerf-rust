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
    pub model: renderer::VolumeRendererConfig,
    pub epoch_count: usize,
    pub train_ratio: f32,
    pub seed: Option<u64>,
}

pub struct Trainer<B: AutodiffBackend> {
    artifact_directory: PathBuf,
    dataset_test: transform::SamplerDataset<
        dataset::SimpleNerfDataset<B>,
        dataset::SimpleNerfDatasetItem,
    >,
    dataset_train: transform::SamplerDataset<
        dataset::SimpleNerfDataset<B>,
        dataset::SimpleNerfDatasetItem,
    >,
    device: B::Device,
    epoch_count: usize,
    learning_rate: f64,
    model: renderer::VolumeRenderer<B>,
    progress_bar: Bar,
    seed: Option<u64>,
}

impl TrainerConfig {
    pub fn init<B: AutodiffBackend<FloatElem = f32>>(
        &self,
        device: &B::Device,
    ) -> Result<Trainer<B>> {
        if let Some(seed) = self.seed {
            B::seed(seed);
        }

        let artifact_directory = PathBuf::from(&self.artifact_directory);

        let (dataset_test, dataset_train) = {
            let datasets = self
                .dataset
                .init_from_file_path_or_url(
                    &self.dataset_file_path_or_url,
                    device,
                )?
                .split_for_training(self.train_ratio);
            let dataset_test_size = datasets.test.len();
            let dataset_train_size = datasets.train.len();
            (
                transform::SamplerDataset::new(
                    datasets.test,
                    dataset_test_size,
                ),
                transform::SamplerDataset::new(
                    datasets.train,
                    dataset_train_size,
                ),
            )
        };

        let model = self.model.init(device)?;

        let progress_bar = {
            let mut bar = tqdm!(
                desc = "Training",
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
            bar.postfix = format!("on {} items", dataset_train.len());
            bar
        };

        let _ = fs::remove_dir_all(&artifact_directory);
        fs::create_dir_all(&artifact_directory)?;
        self.save(artifact_directory.join("config.json"))?;

        Ok(Trainer {
            artifact_directory,
            dataset_train,
            dataset_test,
            device: device.clone(),
            epoch_count: self.epoch_count,
            learning_rate: self.learning_rate,
            model,
            progress_bar,
            seed: self.seed,
        })
    }
}

impl<B: AutodiffBackend<FloatElem = f32>> Trainer<B> {
    pub fn train(&self) -> Result<renderer::VolumeRenderer<B::InnerBackend>> {
        if let Some(seed) = self.seed {
            B::seed(seed);
        }

        let criterion = loss::MseLoss::new();
        let mut model = self.model.clone();
        let mut optimizer = optim::AdamConfig::new().init();
        let mut progress_bar = self.progress_bar.clone();

        term::init(stderr().is_terminal());

        for _ in 0..self.epoch_count {
            let input = self.dataset_train.get(0);
            if input.is_none() {
                break;
            }
            let input = input.unwrap();

            let image_rendered = {
                let directions =
                    Tensor::from_data(input.directions, &self.device);
                let distances =
                    Tensor::from_data(input.distances, &self.device);
                let positions =
                    Tensor::from_data(input.positions, &self.device);
                model.forward(directions, distances, positions)
            };
            let image_true = Tensor::from_data(input.image, &self.device);

            let loss = criterion.forward(
                image_rendered,
                image_true,
                loss::Reduction::Mean,
            );

            let gradients =
                optim::GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(self.learning_rate, model, gradients);

            progress_bar.update(1)?;
        }

        progress_bar.clear()?;
        progress_bar
            .set_bar_format(
                "{desc suffix=''} {postfix} ┃ \
                {total} {unit} ┃ \
                {rate:.1} {unit}/s ┃ \
                {elapsed human=true}\n",
            )
            .map_err(|e| anyhow!(e))?;
        progress_bar.set_description("Trained");
        progress_bar.refresh()?;

        let model_trained = {
            let model = model.valid();
            model.clone().save_file(
                self.artifact_directory.join("model"),
                &record::NamedMpkGzFileRecorder::<record::FullPrecisionSettings>::new(),
            )?;
            model
        };

        Ok(model_trained)
    }
}
