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
        intervals: Tensor<B, 4>,
        positions: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [height, width, points_per_ray, ..] = directions.dims();

        let (colors, densities) = {
            let colors_shape = [height, width, points_per_ray, 3];
            let densities_shape = [height, width, points_per_ray, 1];
            let chunk_count = (((height * width * points_per_ray) as f32)
                / (self.chunk_size as f32))
                .round() as usize;

            let directions_chunks =
                directions.reshape([-1, 3]).chunk(chunk_count, 0);
            let positions_chunks =
                positions.reshape([-1, 3]).chunk(chunk_count, 0);

            let scene_outputs = Tensor::cat(
                directions_chunks
                    .into_iter()
                    .zip(positions_chunks.into_iter())
                    .map(|(directions, positions)| {
                        self.scene.forward(directions, positions)
                    })
                    .collect(),
                0,
            );

            let size = scene_outputs.dims()[0];

            (
                scene_outputs
                    .clone()
                    .slice([0..size, 0..3])
                    .reshape(colors_shape),
                scene_outputs.slice([0..size, 3..4]).reshape(densities_shape),
            )
        };

        let image = {
            let translucency = (-densities * intervals).exp();

            let cumulative_translucency = {
                let mut product = translucency.clone() + 1e-9;

                for index in 1..points_per_ray {
                    product = product.clone().slice_assign(
                        [0..height, 0..width, index..index + 1],
                        product.clone().slice([
                            0..height,
                            0..width,
                            index - 1..index,
                        ]) * product.slice([
                            0..height,
                            0..height,
                            index..index + 1,
                        ]),
                    );
                }

                product
            };

            let transmittance = (-translucency + 1.0) * cumulative_translucency;

            let image = (colors * transmittance).sum_dim(2).squeeze::<3>(2);

            image
        };

        image
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    type Backend = burn::backend::Wgpu;

    #[test]
    fn volume_renderer_output_shape() {
        let device = Default::default();

        let points_per_ray = 16;
        let renderer = VolumeRendererConfig {
            points_per_ray,
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
        let directions = Tensor::random(
            [100, 100, points_per_ray, 3],
            Distribution::Default,
            &device,
        );
        let distances = Tensor::arange(0..points_per_ray as i64, &device)
            .reshape([1, 1, points_per_ray, 1])
            .expand([100, 100, points_per_ray, 1])
            .float()
            + Tensor::random(
                [100, 100, points_per_ray, 1],
                Distribution::Default,
                &device,
            );
        let positions = Tensor::random(
            [100, 100, points_per_ray, 3],
            Distribution::Default,
            &device,
        );

        let outputs = renderer.forward(directions, distances, positions);
        assert_eq!(outputs.dims(), [100, 100, 3]);
    }
}
