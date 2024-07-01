use burn::prelude::*;
use std::f32::consts::PI;

pub struct PositionalEncoderConfig {
    pub encoding_factor: usize,
}

#[derive(Debug, Module)]
pub struct PositionalEncoder<B: Backend> {
    freqs: Tensor<B, 3>,
    phases: Tensor<B, 3>,
}

impl PositionalEncoderConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device
    ) -> Result<PositionalEncoder<B>, String> {
        let encoding_factor = self.encoding_factor;
        if encoding_factor == 0 {
            return Err("Encoding factor must be greater than 0".to_string());
        }

        let levels = Tensor::arange(0..encoding_factor as i64, device);
        let freqs = (
            Tensor::full([encoding_factor], 2, device).powi(levels).float() * PI
        )
            .unsqueeze_dim::<2>(1)
            .repeat(1, 2)
            .reshape([1, encoding_factor * 2, 1]);
        let phases = Tensor::<B, 1>
            ::from_floats([0.0, PI / 2.0], device)
            .unsqueeze_dim::<2>(0)
            .repeat(0, encoding_factor)
            .reshape([1, encoding_factor * 2, 1]);

        Ok(PositionalEncoder {
            freqs: freqs.clone(),
            phases: phases.clone(),
        })
    }

    pub fn get_output_size(&self, input_size: usize) -> usize {
        input_size * (2 * self.encoding_factor + 1)
    }
}

impl<B: Backend> PositionalEncoder<B> {
    pub fn forward(&self, coordinates: Tensor<B, 2>) -> Tensor<B, 2> {
        let coordinates = coordinates.unsqueeze_dim::<3>(1);
        let coordinate_count = coordinates.shape().dims[0];
        let features = (
            coordinates.clone() * self.freqs.clone() +
            self.phases.clone()
        ).sin();
        let features = Tensor::cat(vec![coordinates, features], 1).reshape([
            coordinate_count as i32,
            -1,
        ]);

        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend;
    use burn::tensor::Tensor;

    #[test]
    fn test_output_size() {
        let device = backend::wgpu::WgpuDevice::BestAvailable;

        let config = PositionalEncoderConfig {
            encoding_factor: 10,
        };
        let model = config.init::<backend::Wgpu>(&device);
        assert!(model.is_ok());
        let model = model.unwrap();
        let input = Tensor::from_floats([[1.0, -2.0, 0.0]], &device);
        let output = model.forward(input.clone());
        assert_eq!(
            output.shape().dims[1],
            config.get_output_size(input.shape().dims[1])
        );

        let config = PositionalEncoderConfig {
            encoding_factor: 12,
        };
        let model = config.init::<backend::Wgpu>(&device);
        assert!(model.is_ok());
        let model = model.unwrap();
        let input = Tensor::from_floats([[1.0, -2.5, 0.5, 3.0, -5.5]], &device);
        let output = model.forward(input.clone());
        assert_eq!(
            output.shape().dims[1],
            config.get_output_size(input.shape().dims[1])
        );

        let config_invalid = PositionalEncoderConfig {
            encoding_factor: 0,
        };
        let model = config_invalid.init::<backend::Wgpu>(&device);
        assert!(model.is_err());
    }
}