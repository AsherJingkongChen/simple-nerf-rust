use crate::*;
use anyhow::Result;
use burn::prelude::*;

#[derive(Config, Debug)]
pub struct VolumeRendererConfig {
    pub scene: scene::VolumetricSceneConfig,
}

#[derive(Debug, Module)]
pub struct VolumeRenderer<B: Backend> {
    scene: scene::VolumetricScene<B>,
}

impl VolumeRendererConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<VolumeRenderer<B>> {
        Ok(VolumeRenderer {
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

        let scene_outputs = {
            // NOTE: Using hardset chunk count to be acceptible for Wgpu backend with Metal device
            let chunk_count = 4;

            let directions_chunks =
                directions.reshape([-1, 3]).chunk(chunk_count, 0);
            let positions_chunks =
                positions.reshape([-1, 3]).chunk(chunk_count, 0);

            Tensor::cat(
                directions_chunks
                    .into_iter()
                    .zip(positions_chunks.into_iter())
                    .map(|(directions, positions)| {
                        self.scene.forward(directions, positions)
                    })
                    .collect(),
                0,
            )
            .reshape([height, width, points_per_ray, 4])
        };

        let colors = {
            let indexs = [0..height, 0..width, 0..points_per_ray, 0..3];
            scene_outputs.clone().slice(indexs)
        };

        let densities = {
            let indexs = [0..height, 0..width, 0..points_per_ray, 3..4];
            scene_outputs.slice(indexs)
        };

        let image = {
            let translucency = (-densities * intervals).exp();

            let cumulative_translucency = {
                let mut cumulative_product = translucency.clone() + 1e-9;

                // NOTE: This is a naive implementation of cumulative product
                for index in 1..points_per_ray {
                    let product = cumulative_product.clone().slice([
                        0..height,
                        0..width,
                        index - 1..index,
                    ]) * cumulative_product.clone().slice([
                        0..height,
                        0..width,
                        index..index + 1,
                    ]);

                    cumulative_product = cumulative_product.slice_assign(
                        [0..height, 0..width, index..index + 1],
                        product,
                    );
                }

                cumulative_product
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
