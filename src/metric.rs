use burn::prelude::*;
use std::marker::PhantomData;

#[derive(Clone, Debug)]
pub struct PsnrMetric<B: Backend> {
    coefficient: B::FloatElem,
    _b: PhantomData<B>,
}

impl<B: Backend> PsnrMetric<B> {
    pub fn init(device: &B::Device) -> Self {
        let ten = Tensor::<B, 1>::from_floats([10.0], device);
        let coefficient = (-ten.clone() / ten.log()).into_scalar();
        Self {
            coefficient,
            _b: PhantomData,
        }
    }

    pub fn forward<const D: usize>(
        &self,
        logits: Tensor<B, D>,
        targets: Tensor<B, D>,
    ) -> Tensor<B, 1> {
        let error = logits - targets;
        self.from_mse((error.clone() * error).mean())
    }

    pub fn from_mse(
        &self,
        loss: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        loss.log() * self.coefficient
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Backend = burn::backend::Wgpu;

    #[test]
    fn psnr_metric_output() {
        let device = Default::default();
        let metric = PsnrMetric::<Backend>::init(&device);

        let logits =
            Tensor::<Backend, 2>::from_floats([[0.0, 0.1, 0.2], [0.5, 0.4, 0.3]], &device);
        let targets =
            Tensor::from_floats([[0.5, 0.6, 0.7], [0.0, 0.9, 0.8]], &device);
        let psnr_true = Tensor::<Backend, 1>::from_floats([6.0206003], &device);
        let psnr = metric.forward(logits, targets);
        assert!(psnr.equal(psnr_true).all().into_scalar());

        let logits =
            Tensor::<Backend, 2>::from_floats([[0.0, 0.1, 0.2], [0.5, 0.4, 0.3]], &device);
        let targets =
            Tensor::from_floats([[0.0, 0.6, 0.7], [0.0, 0.4, 0.3]], &device);
        let psnr_true = Tensor::<Backend, 1>::from_floats([9.0309], &device);
        let psnr = metric.forward(logits, targets);
        assert!(psnr.equal(psnr_true).all().into_scalar());
    }
}
