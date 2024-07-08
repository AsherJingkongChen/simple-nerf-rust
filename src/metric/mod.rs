use burn::prelude::*;

#[derive(Clone, Debug)]
pub struct PsnrMetric<B: Backend> {
    log_10: Tensor<B, 1>,
}

impl<B: Backend> PsnrMetric<B> {
    pub fn init(device: &B::Device) -> Self {
        Self {
            log_10: Tensor::from_floats([10.0], device).log(),
        }
    }

    pub fn forward<const D: usize>(
        &self,
        logits: Tensor<B, D>,
        targets: Tensor<B, D>,
    ) -> Tensor<B, 1> {
        (logits - targets).powi_scalar(2).mean().log() / self.log_10.clone()
            * -10.0
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
            Tensor::from_floats([[0.0, 0.1, 0.2], [0.5, 0.4, 0.3]], &device);
        let targets =
            Tensor::from_floats([[0.5, 0.6, 0.7], [0.0, 0.9, 0.8]], &device);
        let psnr_true = Tensor::<Backend, 1>::from_floats([6.0206003], &device);
        let psnr = metric.forward(logits, targets);
        assert!(psnr.equal(psnr_true).all().into_scalar());

        let logits =
            Tensor::from_floats([[0.0, 0.1, 0.2], [0.5, 0.4, 0.3]], &device);
        let targets =
            Tensor::from_floats([[0.0, 0.6, 0.7], [0.0, 0.4, 0.3]], &device);
        let psnr_true = Tensor::<Backend, 1>::from_floats([9.0309], &device);
        let psnr = metric.forward(logits, targets);
        assert!(psnr.equal(psnr_true).all().into_scalar());
    }
}
