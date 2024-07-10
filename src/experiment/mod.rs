pub mod tester;
pub mod trainer;

use crate::*;

use self::{tester::*, trainer::*};
use anyhow::{bail, Result};
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
    pub epoch_count: usize,
    pub learning_rate: f64,
    pub renderer: renderer::VolumeRendererConfig,
    pub train_ratio: f32,
}

pub struct Experiment<B: AutodiffBackend> {
    pub trainer: Trainer<B>,
    pub tester: Tester<B>,
}

impl ExperimentConfig {
    pub fn init<B: AutodiffBackend>(
        &self,
        device: &B::Device,
        do_clear_artifacts_directory: bool,
    ) -> Result<Experiment<B>> {
        let artifact_directory = PathBuf::from(&self.artifact_directory);

        let criterion = loss::MseLoss::new();

        let datasets = self
            .dataset
            .init_from_file_path_or_url(&self.dataset_file_path_or_url, device)?
            .split_for_training(self.train_ratio);

        let metric_fidelity_psnr =
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

        // Prepare the Directory to Save Artifacts
        if do_clear_artifacts_directory {
            if artifact_directory.is_dir() {
                fs::remove_dir_all(&artifact_directory)?;
            } else if artifact_directory.exists() {
                fs::remove_file(&artifact_directory)?;
            }
        } else {
            if artifact_directory.is_dir() {
                bail!(
                    "Artifacts directory already exists: {:?}",
                    artifact_directory
                );
            } else if artifact_directory.exists() {
                bail!(
                    "Artifacts directory is not a directory: {:?}",
                    artifact_directory
                );
            }
        }
        fs::create_dir_all(&artifact_directory)?;

        self.save(artifact_directory.join("experiment.json"))?;

        Ok(Experiment {
            tester: Tester {
                artifact_directory: artifact_directory.clone(),
                dataset: datasets.test,
                device: device.clone(),
                metric_fidelity_psnr: metric_fidelity_psnr.clone(),
            },
            trainer: Trainer {
                artifact_directory,
                criterion,
                dataset: datasets.train,
                device: device.clone(),
                epoch_count: self.epoch_count,
                learning_rate: self.learning_rate,
                metric_fidelity_psnr,
                progress_bar,
                renderer,
            },
        })
    }
}
