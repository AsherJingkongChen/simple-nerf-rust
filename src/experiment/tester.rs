use crate::*;

use anyhow::{anyhow, Result};
use burn::{
    data::dataset::Dataset, prelude::*, tensor::backend::AutodiffBackend,
};
use image::{ImageFormat, RgbImage};
use std::{path::PathBuf, time};

#[derive(Clone, Debug)]
pub struct Tester<B: AutodiffBackend> {
    pub(super) artifact_directory: PathBuf,
    pub(super) dataset: dataset::SimpleNerfDataset<B>,
    pub(super) device: B::Device,
    pub(super) metric_fidelity_psnr: metric::PsnrMetric<B::InnerBackend>,
}

#[derive(Config, Debug)]
pub struct TestOutput {
    pub collage_path: PathBuf,
    pub eval_output: EvaluationOutput,
}

#[derive(Config, Debug)]
pub struct EvaluationOutput {
    pub fps: f64,
    pub items: Vec<EvaluationOutputItem>,
}

#[derive(Config, Debug)]
pub struct EvaluationOutputItem {
    pub index: usize,
    pub fidelity_psnr: f64,
}

impl<B: AutodiffBackend> Tester<B> {
    pub fn test(
        &self,
        renderer: renderer::VolumeRenderer<B::InnerBackend>,
    ) -> Result<TestOutput>
    where
        B::FloatElem: Into<f64>,
    {
        let count = self.dataset.len();
        eprintln!("Testing on {} items", count);

        let mut eval_output_items = vec![];
        let mut input_images = vec![];
        let mut output_images = vec![];
        let mut time_secs_rendering = 0.0;

        // Testing and Evaluating
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

            let fidelity_psnr = self
                .metric_fidelity_psnr
                .forward(output_image.clone(), input.image.clone())
                .into_scalar()
                .into();

            eval_output_items.push(EvaluationOutputItem {
                index,
                fidelity_psnr,
            });
            input_images.push(input.image);
            output_images.push(output_image);

            eprintln!("Item {:03} ┃ PSNR = {:.2} dB", index, fidelity_psnr);
        }

        // Saving the Outputs
        let fps_rendering = count as f64 / time_secs_rendering;
        eprintln!(
            "Rendering time ┃ {:.3} sec ┃ {:.2} FPS",
            time_secs_rendering, fps_rendering
        );

        let eval_output = EvaluationOutput {
            items: eval_output_items,
            fps: fps_rendering,
        };
        eval_output
            .save(&self.artifact_directory.join("evaluation-output.json"))?;

        let collage_path = self.artifact_directory.join("collage.png");
        let collage = {
            let image = Tensor::cat(
                vec![
                    Tensor::cat(input_images, 0),
                    Tensor::cat(output_images, 0),
                ],
                1,
            );
            let [height, width, ..] = image.dims();
            let image = (image.clamp(0.0, 1.0) * 255.0)
                .into_data()
                .convert::<u8>()
                .to_vec::<u8>().map_err(|e| anyhow!("{:?}", e))?;

            RgbImage::from_raw(width as u32, height as u32, image)
                .ok_or(anyhow!("Collage buffer is too small"))?
        };
        collage.save_with_format(&collage_path, ImageFormat::Png)?;
        eprintln!("Collage is saved at {:?}", collage_path);

        Ok(TestOutput {
            collage_path,
            eval_output,
        })
    }
}
