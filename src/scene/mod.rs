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
    output_layer_colors: nn::Linear<B>,
    output_layer_opacities: nn::Linear<B>,
    skip_indexs: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct VolumetricSceneOutput<B: Backend> {
    pub colors: Tensor<B, 2>,
    pub opacities: Tensor<B, 2>,
}

impl VolumetricSceneConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<VolumetricScene<B>> {
        let i = self.input_encoder.get_output_size(6);
        let h = self.hidden_size;
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
            output_layer_colors: nn::LinearConfig::new(h, 3).init(device),
            output_layer_opacities: nn::LinearConfig::new(h, 1).init(device),
            skip_indexs: vec![5],
        })
    }
}

impl<B: Backend> VolumetricScene<B> {
    pub fn forward(
        &self,
        directions: Tensor<B, 2>,
        positions: Tensor<B, 2>,
    ) -> VolumetricSceneOutput<B> {
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
        let colors = activation::sigmoid(
            self.output_layer_colors.forward(features.clone()),
        );
        let opacities =
            activation::relu(self.output_layer_opacities.forward(features));

        VolumetricSceneOutput {
            colors,
            opacities,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    type Backend = burn::backend::Wgpu;

    #[test]
    fn output_shape() {
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

        assert_eq!(outputs.colors.dims(), [123, 3]);
        assert_eq!(outputs.opacities.dims(), [123, 1]);
    }
}
