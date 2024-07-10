pub mod tester;
pub mod trainer;

use crate::*;

use self::{tester::*, trainer::*};
use anyhow::Result;
use burn::{
    data::dataset::Dataset, nn::loss, prelude::*,
    tensor::backend::AutodiffBackend,
};
use kdam::tqdm;
use std::{fs, path::PathBuf};

#[derive(Config, Debug)]
pub struct ExperimentConfig {
    pub artifact_directory: String,
    pub dataset: dataset::SimpleNerfDatasetConfig,
    pub dataset_file_path_or_url: String,
    pub learning_rate: f64,
    pub epoch_count: usize,
    pub train_ratio: f32,
    pub renderer: renderer::VolumeRendererConfig,
}

pub struct Experiment<B: AutodiffBackend> {
    pub trainer: Trainer<B>,
    pub tester: Tester<B>,
}

impl ExperimentConfig {
    pub fn init<B: AutodiffBackend>(
        &self,
        device: &B::Device,
    ) -> Result<Experiment<B>> {
        let artifact_directory = PathBuf::from(&self.artifact_directory);

        let criterion = loss::MseLoss::new();

        let datasets = self
            .dataset
            .init_from_file_path_or_url(&self.dataset_file_path_or_url, device)?
            .split_for_training(self.train_ratio);

        let metric_fidelity =
            metric::PsnrMetric::<B::InnerBackend>::init(device);

        let renderer = self.renderer.init(device)?;

        let progress_bar = {
            let mut bar = tqdm!(
                desc = format!("Training on {} items", datasets.train.len()),
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
            bar.postfix = "┃ PSNR = 0.00 dB".into();
            bar
        };

        let _ = fs::remove_dir_all(&artifact_directory);
        fs::create_dir_all(&artifact_directory)?;
        self.save(artifact_directory.join("config.json"))?;

        Ok(Experiment {
            tester: Tester {
                artifact_directory: artifact_directory.clone(),
                dataset: datasets.test,
                device: device.clone(),
                metric_fidelity: metric_fidelity.clone(),
            },
            trainer: Trainer {
                artifact_directory,
                criterion,
                dataset: datasets.train,
                device: device.clone(),
                epoch_count: self.epoch_count,
                learning_rate: self.learning_rate,
                metric_fidelity,
                renderer,
                progress_bar,
            },
        })
    }
}
