use crate::*;
use anyhow::Result;
use burn::{prelude::*, tensor::activation};

#[derive(Config, Debug)]
pub struct VolumetricSceneConfig {
    pub hidden_size: usize,
    pub input_encoder: encoder::PositionalEncoderConfig,
}

#[derive(Debug, Module)]
pub struct VolumetricScene<B: Backend> {
    input_encoder: encoder::PositionalEncoder<B>,
    hidden_layers: Vec<nn::Linear<B>>,
    output_layer: nn::Linear<B>,
    skip_indexs: Vec<usize>,
}

impl VolumetricSceneConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<VolumetricScene<B>> {
        let i = self.input_encoder.get_output_size(6);
        let h = self.hidden_size;
        let o = 3 + 1;
        Ok(VolumetricScene {
            input_encoder: self.input_encoder.init(device)?,
            hidden_layers: vec![
                nn::LinearConfig::new(i, h).init(device),
                nn::LinearConfig::new(h, h).init(device),
                nn::LinearConfig::new(h, h).init(device),
                nn::LinearConfig::new(h, h).init(device),
                nn::LinearConfig::new(h, h).init(device),
                nn::LinearConfig::new(h + i, h).init(device),
                nn::LinearConfig::new(h, h).init(device),
                nn::LinearConfig::new(h, h).init(device),
            ],
            output_layer: nn::LinearConfig::new(h, o).init(device),
            skip_indexs: vec![5],
        })
    }
}

impl<B: Backend> VolumetricScene<B> {
    pub fn forward(
        &self,
        directions: Tensor<B, 2>,
        positions: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let inputs = self
            .input_encoder
            .forward(Tensor::cat(vec![directions, positions], 1));
        let mut features = inputs.clone();

        for (index, layer) in self.hidden_layers.iter().enumerate() {
            if self.skip_indexs.contains(&index) {
                features = Tensor::cat(vec![features, inputs.clone()], 1);
            }
            features = layer.forward(features);
            features = activation::relu(features);
        }

        let outputs = {
            features = self.output_layer.forward(features);
            let size = features.dims()[0];
            let colors =
                activation::sigmoid(features.clone().slice([0..size, 0..3]));
            let densities = activation::relu(features.slice([0..size, 3..4]));
            Tensor::cat(vec![colors, densities], 1)
        };

        outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    type Backend = burn::backend::Wgpu;

    #[test]
    fn volumetric_scene_output_shape() {
        let config = VolumetricSceneConfig {
            hidden_size: 1,
            input_encoder: encoder::PositionalEncoderConfig {
                encoding_factor: 1,
            },
        };
        let device = Default::default();

        let model = config.init::<Backend>(&device).unwrap();

        let positions =
            Tensor::random([123, 3], Distribution::Default, &device);
        let directions = positions.random_like(Distribution::Default);

        let outputs = model.forward(positions, directions);
        assert_eq!(outputs.dims(), [123, 4]);
    }
}
