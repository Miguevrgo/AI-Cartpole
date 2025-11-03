use ndarray::{Array1, Array2};
use rand::Rng;
use std::ops::Range;

struct Layer {
    size: usize,
    weights: Array2<f32>,
    bias: Array1<f32>,
}

impl Layer {
    pub fn new(size: usize, input_size: usize, range: Range<f64>) -> Self {
        let mut rng = rand::rng();

        let weights = Array2::from_shape_simple_fn((size, input_size), || {
            rng.random_range(range.clone()) as f32
        });
        let bias = Array1::zeros(size);

        Layer {
            size,
            weights,
            bias,
        }
    }
}

struct Network {
    num_layers: usize,
    layers: Array1<Layer>,
}

impl Network {
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        let mut layers = Vec::with_capacity(layer_sizes.len());
        let mut prev_size = 4; // Number of fields in Cartpole
        for &size in &layer_sizes {
            layers.push(Layer::new(size, prev_size, 0.0..1.0));
            prev_size = size;
        }

        Self {
            num_layers: layer_sizes.len(),
            layers: Array1::from_vec(layers),
        }
    }
}
