extern crate simple_nerf;

use simple_nerf::prelude::*;

fn main() -> anyhow::Result<()> {
    type InnerBackend = backend::Wgpu;
    type Backend = backend::Autodiff<InnerBackend>;

    let device = Default::default();

    let experiment = experiment::ExperimentConfig {
        artifact_directory: "artifacts/experiment".into(),
        dataset: dataset::SimpleNerfDatasetConfig {
            points_per_ray: 20,
            distance_range: 2.0..6.0,
        },
        dataset_file_path_or_url: "resources/lego-tiny/data.npz".into(),
        epoch_count: 10000,
        learning_rate: 1e-3,
        renderer: renderer::VolumeRendererConfig {
            scene: scene::VolumetricSceneConfig {
                hidden_size: 256,
                input_encoder: encoder::PositionalEncoderConfig {
                    encoding_factor: 10,
                },
            },
        },
        train_ratio: 0.8,
    }
    .init::<Backend>(&device, true)?;

    experiment.tester.test(experiment.trainer.train()?)?;

    Ok(())
}
