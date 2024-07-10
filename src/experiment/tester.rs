use crate::*;

use anyhow::Result;
use burn::{
    data::dataset::Dataset, prelude::*, tensor::backend::AutodiffBackend,
};
use std::{path::PathBuf, time, vec};

#[derive(Clone, Debug)]
pub struct Tester<B: AutodiffBackend> {
    pub(super) artifact_directory: PathBuf,
    pub(super) dataset: dataset::SimpleNerfDataset<B>,
    pub(super) device: B::Device,
    pub(super) metric_fidelity: metric::PsnrMetric<B::InnerBackend>,
}

#[derive(Config, Debug)]
pub struct EvaluationOutput {
    pub fps: f64,
    pub items: Vec<EvaluationOutputItem>,
}

#[derive(Config, Debug)]
pub struct EvaluationOutputItem {
    pub index: usize,
    pub fidelity: f64,
}

impl<B: AutodiffBackend> Tester<B> {
    pub fn test(
        &self,
        renderer: renderer::VolumeRenderer<B::InnerBackend>,
    ) -> Result<EvaluationOutput>
    where
        B::FloatElem: Into<f64>,
    {
        let count = self.dataset.len();
        let mut eval_items = vec![];
        let mut time_secs_rendering = 0.0;

        eprintln!("Testing on {} items", count);

        for (index, data) in self.dataset.iter().enumerate() {
            let timer_from_input_to_output = time::Instant::now();

            let input = data.into_input(&self.device);
            let output_image = renderer.forward(
                input.directions,
                input.intervals,
                input.positions,
            );

            time_secs_rendering +=
                timer_from_input_to_output.elapsed().as_secs_f64();

            let fidelity = self
                .metric_fidelity
                .forward(output_image, input.image)
                .into_scalar()
                .into();

            let eval_item = EvaluationOutputItem {
                index,
                fidelity,
            };
            eval_items.push(eval_item);

            eprintln!("Item {:03} ┃ PSNR = {:.2} dB", index, fidelity);
        }

        let fps_rendering = count as f64 / time_secs_rendering;
        eprintln!(
            "Rendering time ┃ {:.3} sec ┃ {:.2} FPS",
            time_secs_rendering, fps_rendering
        );

        let eval_output = EvaluationOutput {
            items: eval_items,
            fps: fps_rendering,
        };

        eval_output
            .save(&self.artifact_directory.join("evaluation-output.json"))?;

        Ok(eval_output)
    }
}
