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
use kdam::{term, Bar, BarExt};
use std::{
    io::{stderr, IsTerminal},
    path::PathBuf,
};

#[derive(Clone, Debug)]
pub struct Trainer<B: AutodiffBackend> {
    pub(super) artifact_directory: PathBuf,
    pub(super) criterion: loss::MseLoss<B>,
    pub(super) dataset: dataset::SimpleNerfDataset<B>,
    pub(super) device: B::Device,
    pub(super) epoch_count: usize,
    pub(super) learning_rate: f64,
    pub(super) metric_fidelity_psnr: metric::PsnrMetric<B::InnerBackend>,
    pub(super) progress_bar: Bar,
    pub(super) renderer: renderer::VolumeRenderer<B>,
}

impl<B: AutodiffBackend> Trainer<B> {
    pub fn train(&self) -> Result<renderer::VolumeRenderer<B::InnerBackend>> {
        let input_profile =
            self.dataset.get(0).map(|data| data.into_input(&self.device));

        let dataset_size = self.dataset.len();
        let dataset =
            transform::SamplerDataset::new(self.dataset.clone(), dataset_size);
        let mut optimizer = optim::AdamConfig::new().init();
        let mut progress_bar = self.progress_bar.clone();
        let mut renderer = self.renderer.clone();

        // Initializing the Progress Bar
        term::init(stderr().is_terminal());
        progress_bar.reset(None);

        // Training
        for epoch in 0..self.epoch_count {
            let input = {
                let data = dataset.get(0);
                if data.is_none() {
                    break;
                }
                data.unwrap().into_input(&self.device)
            };

            let output_image = renderer.forward(
                input.directions,
                input.intervals,
                input.positions,
            );

            let loss = self.criterion.forward(
                output_image,
                input.image,
                loss::Reduction::Mean,
            );

            let gradients =
                optim::GradientsParams::from_grads(loss.backward(), &renderer);
            renderer = optimizer.step(self.learning_rate, renderer, gradients);

            // Profiling
            if input_profile.is_some() && epoch % 25 == 0 {
                let input = input_profile.clone().unwrap();

                let output_image = renderer.valid().forward(
                    input.directions,
                    input.intervals,
                    input.positions,
                );

                let fidelity_psnr = self
                    .metric_fidelity_psnr
                    .forward(output_image, input.image)
                    .into_scalar();
                progress_bar.postfix = format!("┃ PSNR = {:.2} dB", fidelity_psnr);
            }

            progress_bar.update(1)?;
        }

        // Terminating the Progress Bar
        {
            progress_bar.clear()?;
            progress_bar
                .set_bar_format(
                    "{desc suffix=''} ┃ \
                    {total} {unit} ┃ \
                    {rate:.1} {unit}/s ┃ \
                    {elapsed human=true}\n",
                )
                .map_err(|e| anyhow!(e))?;
            progress_bar
                .set_description(format!("Trained on {} items", dataset.len()));
            progress_bar.refresh()?;
        }

        // Saving the Renderer
        renderer.clone().save_file(
            self.artifact_directory.join("volume-renderer"),
            &record::DefaultRecorder::new(),
        )?;

        Ok(renderer.valid())
    }
}
