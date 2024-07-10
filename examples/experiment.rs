extern crate simple_nerf;

use simple_nerf::prelude::*;

fn main() -> anyhow::Result<()> {
    type InnerBackend = backend::Wgpu;
    type Backend = backend::Autodiff<InnerBackend>;

    let device = backend::wgpu::WgpuDevice::BestAvailable;

    let experiment = experiment::ExperimentConfig {
        artifact_directory: "artifacts/experiment".into(),
        dataset: dataset::SimpleNerfDatasetConfig {
            points_per_ray: 10,
            distance_range: 2.0..6.0,
        },
        dataset_file_path_or_url: "resources/lego-tiny/data.npz".into(),
        learning_rate: 5e-4,
        epoch_count: 10,
        train_ratio: 0.8,
        renderer: renderer::VolumeRendererConfig {
            scene: scene::VolumetricSceneConfig {
                hidden_size: 256,
                input_encoder: encoder::PositionalEncoderConfig {
                    encoding_factor: 10,
                },
            },
        },
    }
    .init::<Backend>(&device, true)?;

    let renderer = experiment.trainer.train()?;
    experiment.tester.test(renderer)?;

    Ok(())
}
