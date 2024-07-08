use crate::*;
use anyhow::{bail, Result};
use burn::prelude::*;

#[derive(Config, Debug)]
pub struct VolumeRendererConfig {
    pub points_per_ray: usize,
    pub rays_per_chunk: usize,
    pub scene: scene::VolumetricSceneConfig,
}

#[derive(Debug, Module)]
pub struct VolumeRenderer<B: Backend> {
    chunk_size: usize,
    scene: scene::VolumetricScene<B>,
}

impl VolumeRendererConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<VolumeRenderer<B>> {
        if self.points_per_ray == 0 {
            bail!("Point count per ray must be greater than 0");
        }
        if self.rays_per_chunk == 0 {
            bail!("Ray count per chunk must be greater than 0");
        }
        Ok(VolumeRenderer {
            chunk_size: self.points_per_ray * self.rays_per_chunk,
            scene: self.scene.init(device)?,
        })
    }
}

impl<B: Backend> VolumeRenderer<B> {
    pub fn forward(
        &self,
        directions: Tensor<B, 4>,
        distances: Tensor<B, 4>,
        positions: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [height, width, points_per_ray, ..] = directions.dims();

        let scene_outputs_chunks = {
            let chunk_count = (((height * width * points_per_ray) as f32)
                / (self.chunk_size as f32))
                .round() as usize;
            let directions_chunks =
                directions.flatten::<2>(0, 2).chunk(chunk_count, 0);
            let positions_chunks =
                positions.flatten::<2>(0, 2).chunk(chunk_count, 0);
            directions_chunks.into_iter().zip(positions_chunks.into_iter()).map(
                |(directions, positions)| {
                    self.scene.forward(directions, positions)
                },
            )
        };

        let opacities = {
            let shape = [height, width, points_per_ray, 1];
            Tensor::cat(
                scene_outputs_chunks
                    .clone()
                    .map(|output| output.opacities)
                    .collect(),
                0,
            )
            .reshape(shape)
        };

        let colors = {
            let shape = [height, width, points_per_ray, 3];
            Tensor::cat(
                scene_outputs_chunks.map(|output| output.colors).collect(),
                0,
            )
            .reshape(shape)
        };

        let planar_rgb = {
            let intervals = {
                let device = distances.device();
                let intervals = distances.clone().slice([
                    0..height,
                    0..width,
                    1..points_per_ray,
                ]) - distances.slice([
                    0..height,
                    0..width,
                    0..(points_per_ray - 1),
                ]);
                Tensor::cat(
                    vec![
                        intervals,
                        Tensor::full([height, width, 1, 1], 1e9, &device),
                    ],
                    2,
                )
            };
            let translucency = (-opacities * intervals).exp();
            let transmittance = (-translucency.clone() + 1.0)
                * (translucency + 1e-6).prod_dim(2);

            (colors * transmittance).sum_dim(2).squeeze::<3>(2)
        };

        planar_rgb
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    type Backend = burn::backend::Wgpu;

    #[test]
    fn output_shape() {
        let device = Default::default();
        let renderer = VolumeRendererConfig {
            points_per_ray: 16,
            rays_per_chunk: 250,
            scene: scene::VolumetricSceneConfig {
                input_encoder: encoder::PositionalEncoderConfig {
                    encoding_factor: 3,
                },
                hidden_size: 8,
            },
        }
        .init::<Backend>(&device);
        assert!(renderer.is_ok(), "Error: {}", renderer.unwrap_err());

        let renderer = renderer.unwrap();
        let directions = Tensor::<burn::backend::Wgpu, 4>::random(
            [100, 100, 16, 3],
            Distribution::Default,
            &device,
        );
        let distances = Tensor::<burn::backend::Wgpu, 4>::random(
            [100, 100, 16, 1],
            Distribution::Default,
            &device,
        );
        let positions = Tensor::<burn::backend::Wgpu, 4>::random(
            [100, 100, 16, 3],
            Distribution::Default,
            &device,
        );

        let output = renderer.forward(directions, distances, positions);
        assert_eq!(output.dims(), [100, 100, 3]);
    }
}
