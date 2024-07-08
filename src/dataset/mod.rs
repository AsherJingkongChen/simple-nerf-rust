use burn::{data::dataset::Dataset, prelude::*, tensor::Distribution};
use npyz::{npz, NpyFile};
use regex::Regex;
use reqwest::IntoUrl;
use std::{fs::File, io, ops::Range, path::Path};
use zip::ZipArchive;

#[derive(Config, Debug)]
pub struct SimpleNerfDatasetConfig {
    pub points_per_ray: usize,
    pub distance_range: Range<f32>,
}

#[derive(Clone, Debug)]
pub struct SimpleNerfDataset<B: Backend> {
    device: B::Device,
    inners: Vec<SimpleNerfDatasetInner>,
    pub has_noisy_distance: bool,
}

#[derive(Clone, Debug)]
struct SimpleNerfDatasetInner {
    directions: Data<f32, 4>,
    distances: Data<f32, 4>,
    image: Data<f32, 3>,
    origins: Data<f32, 4>,
}

#[derive(Clone, Debug)]
pub struct SimpleNerfDatasetItem {
    pub directions: Data<f32, 4>,
    pub distances: Data<f32, 4>,
    pub image: Data<f32, 3>,
    pub positions: Data<f32, 4>,
}

#[derive(Clone, Debug)]
pub struct SimpleNerfDatasetBatch<B: Backend> {
    pub directions: Tensor<B, 4>,
    pub distances: Tensor<B, 4>,
    pub image: Tensor<B, 3>,
    pub positions: Tensor<B, 4>,
}

#[derive(Clone, Debug)]
pub struct SimpleNerfDatasetSplit<B: Backend> {
    pub train: SimpleNerfDataset<B>,
    pub test: SimpleNerfDataset<B>,
}

impl SimpleNerfDatasetConfig {
    pub fn init_from_reader<B: Backend, R: io::Read + io::Seek>(
        &self,
        reader: R,
        device: &B::Device,
    ) -> io::Result<SimpleNerfDataset<B>> {
        let points_per_ray = self.points_per_ray;
        if points_per_ray == 0 {
            return Err(io::ErrorKind::InvalidData.into());
        }

        let distance_range = ({
            if self.distance_range.start == self.distance_range.end {
                Err(io::ErrorKind::InvalidData)
            } else if self.distance_range.end < self.distance_range.start {
                Ok(self.distance_range.end..self.distance_range.start)
            } else {
                Ok(self.distance_range.clone())
            }
        })?;

        let mut archive = ZipArchive::new(reader)?;

        let focal = *NpyFile::new(io::BufReader::new(
            archive.by_name(&npz::file_name_from_array_name("focal"))?,
        ))?
        .into_vec::<f64>()?
        .get(0)
        .ok_or(io::ErrorKind::InvalidData)? as f32;

        let images = {
            let array = NpyFile::new(io::BufReader::new(
                archive.by_name(&npz::file_name_from_array_name("images"))?,
            ))?;
            let shape = Shape::from(array.shape().to_vec());
            Tensor::<B, 4>::from_data(
                Data::new(array.into_vec::<f32>()?, shape).convert(),
                device,
            )
        };

        let poses = {
            let array = NpyFile::new(io::BufReader::new(
                archive.by_name(&npz::file_name_from_array_name("poses"))?,
            ))?;
            let shape = Shape::from(array.shape().to_vec());
            Tensor::<B, 3>::from_data(
                Data::new(array.into_vec::<f32>()?, shape).convert(),
                device,
            )
        };

        let [image_count, height, width, channel_count] = images.dims();
        let pose_count = poses.dims()[0];
        if image_count != pose_count {
            return Err(io::ErrorKind::InvalidData.into());
        }
        if channel_count != 3 {
            return Err(io::ErrorKind::InvalidData.into());
        }

        let planes = {
            let planes_shape = [1, height, width, 1, 3];
            let plane_shape = [height, width];
            let plane_x = (Tensor::arange(0..width as i64, device)
                .float()
                .unsqueeze_dim::<2>(0)
                .expand(plane_shape)
                - (width as f32) / 2.0)
                / focal;
            let plane_y = (-Tensor::arange(0..height as i64, device)
                .float()
                .unsqueeze_dim::<2>(1)
                .expand(plane_shape)
                + (height as f32) / 2.0)
                / focal;
            let plane_z = Tensor::full(plane_shape, -1.0, device);
            Tensor::<B, 2>::stack::<3>(vec![plane_x, plane_y, plane_z], 2)
                .reshape(planes_shape)
        };

        let directions = (planes
            * poses
                .clone()
                .slice([0..image_count, 0..3, 0..3])
                .unsqueeze_dims::<5>(&[1, 1]))
        .sum_dim(4)
        .swap_dims(4, 3);

        let origins = poses
            .slice([0..image_count, 0..3, 3..4])
            .unsqueeze_dims::<5>(&[1, 1])
            .swap_dims(4, 3)
            .expand(directions.shape());

        let directions = directions.repeat(3, points_per_ray);

        let distances =
            (Tensor::<B, 1, Int>::arange(0..points_per_ray as i64, device)
                .float()
                * ((distance_range.end - distance_range.start)
                    / (points_per_ray as f32))
                + distance_range.start)
                .unsqueeze_dims::<5>(&[0, 0, 0, -1])
                .expand([image_count, height, width, points_per_ray, 1]);

        let inners = directions
            .iter_dim(0)
            .zip(distances.iter_dim(0))
            .zip(images.iter_dim(0))
            .zip(origins.iter_dim(0))
            .map(|(((directions, distances), image), origins)| {
                SimpleNerfDatasetInner {
                    directions: directions
                        .squeeze::<4>(0)
                        .into_data()
                        .convert(),
                    distances: distances.squeeze::<4>(0).into_data().convert(),
                    image: image.squeeze::<3>(0).into_data().convert(),
                    origins: origins.squeeze::<4>(0).into_data().convert(),
                }
            })
            .collect();

        Ok(SimpleNerfDataset {
            device: device.clone(),
            inners,
            has_noisy_distance: false,
        })
    }

    pub fn init_from_file_path<B: Backend>(
        &self,
        file_path: impl AsRef<Path>,
        device: &B::Device,
    ) -> io::Result<SimpleNerfDataset<B>> {
        self.init_from_reader(File::open(file_path)?, device)
    }

    pub fn init_from_url<B: Backend>(
        &self,
        url: impl IntoUrl,
        device: &B::Device,
    ) -> io::Result<SimpleNerfDataset<B>> {
        self.init_from_reader(
            io::Cursor::new(
                reqwest::blocking::get(url)
                    .or(Err(io::ErrorKind::ConnectionRefused))?
                    .error_for_status()
                    .or(Err(io::ErrorKind::NotFound))?
                    .bytes()
                    .or(Err(io::ErrorKind::Interrupted))?,
            ),
            device,
        )
    }

    pub fn init_from_file_path_or_url<B: Backend>(
        &self,
        file_path_or_url: &str,
        device: &B::Device,
    ) -> io::Result<SimpleNerfDataset<B>> {
        if Regex::new(r"https?://").unwrap().is_match(file_path_or_url) {
            self.init_from_url(file_path_or_url, device)
        } else {
            self.init_from_file_path(file_path_or_url, device)
        }
    }
}

impl<B: Backend> SimpleNerfDataset<B> {
    pub fn split_for_training(
        self,
        ratio: f32,
    ) -> SimpleNerfDatasetSplit<B> {
        let (inners_left, inners_right) = self.inners.split_at(
            (ratio.clamp(0.0, 1.0) * (self.inners.len() as f32)).round()
                as usize,
        );

        SimpleNerfDatasetSplit {
            train: SimpleNerfDataset {
                device: self.device.clone(),
                inners: inners_left.into(),
                has_noisy_distance: true,
            },
            test: SimpleNerfDataset {
                device: self.device,
                inners: inners_right.to_vec(),
                has_noisy_distance: false,
            },
        }
    }
}

impl<B: Backend> Dataset<SimpleNerfDatasetItem> for SimpleNerfDataset<B> {
    fn len(&self) -> usize {
        self.inners.len()
    }

    fn get(
        &self,
        index: usize,
    ) -> Option<SimpleNerfDatasetItem> {
        let inner = self.inners.get(index)?.clone();

        let distances = {
            let distance = {
                let values =
                    inner.distances.value.get(0..2).unwrap_or(&[0.0, 0.0]);
                values[1] - values[0]
            };
            let mut distances =
                Tensor::from_data(inner.distances.convert(), &self.device);
            if self.has_noisy_distance {
                let noises = distances
                    .random_like(Distribution::Uniform(0.0, distance as f64));
                distances = distances + noises;
            }
            distances
        };

        let positions = {
            let positions: Tensor<B, 4> =
                Tensor::from_data(inner.origins.convert(), &self.device)
                    + Tensor::from_data(
                        inner.directions.clone().convert(),
                        &self.device,
                    ) * distances.clone();
            positions.into_data().convert()
        };

        let distances = distances.into_data().convert();

        Some(SimpleNerfDatasetItem {
            directions: inner.directions,
            distances,
            image: inner.image,
            positions,
        })
    }
}

impl<B: Backend> SimpleNerfDatasetBatch<B> {
    pub fn from_item(
        item: SimpleNerfDatasetItem,
        device: &B::Device,
    ) -> SimpleNerfDatasetBatch<B> {
        SimpleNerfDatasetBatch {
            directions: Tensor::from_data(item.directions.convert(), device),
            distances: Tensor::from_data(item.distances.convert(), device),
            image: Tensor::from_data(item.image.convert(), device),
            positions: Tensor::from_data(item.positions.convert(), device),
        }
    }
}

impl SimpleNerfDatasetItem {
    pub fn into_batch<B: Backend>(
        self,
        device: &B::Device,
    ) -> SimpleNerfDatasetBatch<B> {
        SimpleNerfDatasetBatch::from_item(self, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Backend = burn::backend::Wgpu;

    const TEST_DATA_FILE_PATH: &str = "resources/lego-tiny/data.npz";
    const TEST_DATA_URL: &str =
        "https://raw.githubusercontent.com/AsherJingkongChen/simple-nerf-rust/main/resources/lego-tiny/data.npz";

    #[test]
    fn simple_nerf_dataset_output_shape() {
        let device = Default::default();

        let dataset = (SimpleNerfDatasetConfig {
            points_per_ray: 7,
            distance_range: 2.0..6.0,
        })
        .init_from_file_path::<Backend>(TEST_DATA_FILE_PATH, &device);
        assert!(dataset.is_ok(), "Error: {}", dataset.unwrap_err());

        let dataset = dataset.unwrap();
        let item = dataset.get(0);
        assert!(item.is_some());

        let item = item.unwrap();
        assert_eq!(item.directions.shape.dims, [100, 100, 7, 3]);
        assert_eq!(item.distances.shape.dims, [100, 100, 7, 1]);
        assert_eq!(item.image.shape.dims, [100, 100, 3]);
        assert_eq!(item.positions.shape.dims, [100, 100, 7, 3]);
        assert_eq!(item.positions.shape.dims, item.directions.shape.dims);

        let inners = dataset.inners;
        assert_eq!(inners.len(), 106);

        let inner = inners.get(0);
        assert!(inner.is_some());

        let inner = inner.unwrap();
        assert_eq!(inner.directions.shape.dims, [100, 100, 7, 3]);
        assert_eq!(inner.distances.shape.dims, [100, 100, 7, 1]);
        assert_eq!(inner.image.shape.dims, [100, 100, 3]);
        assert_eq!(inner.origins.shape.dims, [100, 100, 1, 3]);
    }

    #[test]
    fn simple_nerf_dataset_remote_retrieval() {
        let device = Default::default();

        let dataset = (SimpleNerfDatasetConfig {
            points_per_ray: 7,
            distance_range: 2.0..6.0,
        })
        .init_from_url::<Backend>(TEST_DATA_URL, &device);
        assert!(dataset.is_ok(), "Error: {}", dataset.unwrap_err());

        let dataset = dataset.unwrap();
        assert_eq!(dataset.inners.len(), 106);
    }

    #[test]
    fn simple_nerf_dataset_splitting() {
        let device = Default::default();

        let dataset = (SimpleNerfDatasetConfig {
            points_per_ray: 8,
            distance_range: 2.0..6.0,
        })
        .init_from_file_path::<Backend>(TEST_DATA_FILE_PATH, &device);
        assert!(dataset.is_ok(), "Error: {}", dataset.unwrap_err());

        let dataset = dataset.unwrap();
        let datasets = dataset.split_for_training(0.8);
        assert_eq!(datasets.train.len(), datasets.train.inners.len());
        assert_eq!(datasets.test.len(), datasets.test.inners.len());
        assert!(datasets.train.has_noisy_distance);
        assert!(!datasets.test.has_noisy_distance);

        let datasets = datasets.train.split_for_training(1.0);
        assert_eq!(datasets.train.len(), datasets.train.inners.len());
        assert_eq!(datasets.test.len(), datasets.test.inners.len());
        assert_eq!(datasets.test.len(), 0);
        assert!(datasets.train.has_noisy_distance);
        assert!(!datasets.test.has_noisy_distance);
    }
}
