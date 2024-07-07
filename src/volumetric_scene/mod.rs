use crate::*;
use burn::{ prelude::*, tensor::activation };

#[derive(Config, Debug)]
pub struct VolumetricSceneConfig {
    pub hidden_size: usize,
    pub input_encoder: positional_encoder::PositionalEncoderConfig,
}

#[derive(Debug, Module)]
pub struct VolumetricScene<B: Backend> {
    input_encoder: positional_encoder::PositionalEncoder<B>,
    hidden_layers: Vec<nn::Linear<B>>,
    output_layer: nn::Linear<B>,
    skip_indexs: Vec<usize>,
}

impl VolumetricSceneConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device
    ) -> Result<VolumetricScene<B>, String> {
        let i = self.input_encoder.get_output_size(6);
        let h = self.hidden_size;
        let o = 4;
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
                nn::LinearConfig::new(h, h).init(device)
            ],
            output_layer: nn::LinearConfig::new(h, o).init(device),
            skip_indexs: vec![5],
        })
    }
}

impl<B: Backend> VolumetricScene<B> {
    pub fn forward(
        &self,
        positions: Tensor<B, 2>,
        directions: Tensor<B, 2>
    ) -> Tensor<B, 2> {
        let inputs = self.input_encoder.forward(
            Tensor::cat(vec![positions, directions], 1)
        );
        let size = inputs.dims()[0];
        let mut outputs = inputs.clone();
        for (index, layer) in self.hidden_layers.iter().enumerate() {
            if self.skip_indexs.contains(&index) {
                outputs = Tensor::cat(vec![outputs, inputs.clone()], 1);
            }
            outputs = layer.forward(outputs);
            outputs = activation::relu(outputs);
        }
        outputs = self.output_layer.forward(outputs);
        outputs = Tensor::cat(
            vec![
                activation::sigmoid(outputs.clone().slice([0..size, 0..3])),
                activation::relu(outputs.slice([0..size, 3..4]))
            ],
            1
        );
        outputs
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
            input_encoder: positional_encoder::PositionalEncoderConfig {
                encoding_factor: 1,
            },
        };
        let device = Default::default();
        let model = config.init::<Backend>(&device).unwrap();
        let positions = Tensor::random(
            [123, 3],
            Distribution::Default,
            &device
        );
        let directions = positions.random_like(Distribution::Default);
        let outputs = model.forward(positions, directions);
        assert_eq!(outputs.dims(), [123, 4]);
    }
}
