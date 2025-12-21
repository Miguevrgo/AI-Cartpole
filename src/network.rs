use ndarray::{Array1, Array2};
use rand::Rng;

use crate::constants::LEARNING_RATE;

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn relu_derivative(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

fn clip_value(x: f32, min: f32, max: f32) -> f32 {
    x.max(min).min(max)
}

pub struct Layer {
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
    last_input: Option<Array1<f32>>,
    last_output_pre_activation: Option<Array1<f32>>,
}

impl Layer {
    pub fn new(output_size: usize, input_size: usize) -> Self {
        let mut rng = rand::rng();
        let scale = (2.0 / input_size as f32).sqrt();

        let weights = Array2::from_shape_simple_fn((output_size, input_size), || {
            rng.random_range(-scale..scale)
        });
        let bias = Array1::zeros(output_size);

        Layer {
            weights,
            bias,
            last_input: None,
            last_output_pre_activation: None,
        }
    }

    pub fn forward(&mut self, input: &Array1<f32>, use_activation: bool) -> Array1<f32> {
        self.last_input = Some(input.clone());
        let output = self.weights.dot(input) + &self.bias;
        self.last_output_pre_activation = Some(output.clone());

        if use_activation {
            output.mapv(relu)
        } else {
            output
        }
    }

    pub fn backward(&mut self, grad_output: &Array1<f32>, use_activation: bool) -> Array1<f32> {
        let grad = if use_activation {
            let pre_activation = self.last_output_pre_activation.as_ref().unwrap();
            grad_output * &pre_activation.mapv(relu_derivative)
        } else {
            grad_output.clone()
        };

        let grad = grad.mapv(|x| clip_value(x, -1.0, 1.0));

        let input = self.last_input.as_ref().unwrap();

        let grad_weights = grad
            .clone()
            .insert_axis(ndarray::Axis(1))
            .dot(&input.clone().insert_axis(ndarray::Axis(0)));
        let grad_bias = grad.clone();
        let grad_input = self.weights.t().dot(&grad);

        self.weights = &self.weights - &(grad_weights * LEARNING_RATE);
        self.bias = &self.bias - &(grad_bias * LEARNING_RATE);

        grad_input
    }
}

impl Clone for Layer {
    fn clone(&self) -> Self {
        Layer {
            weights: self.weights.clone(),
            bias: self.bias.clone(),
            last_input: None,
            last_output_pre_activation: None,
        }
    }
}

pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        let mut prev_size = 4;

        for &size in layer_sizes {
            layers.push(Layer::new(size, prev_size));
            prev_size = size;
        }

        Self { layers }
    }

    pub fn forward(&mut self, input: &[f32; 4]) -> Array1<f32> {
        let mut output = Array1::from_vec(input.to_vec());

        let num_layers = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let use_activation = i < num_layers - 1;
            output = layer.forward(&output, use_activation);
        }

        output
    }

    pub fn backward(&mut self, target: &Array1<f32>, prediction: &Array1<f32>) {
        let mut grad = prediction - target;
        let num_layers = self.layers.len();

        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            let use_activation = i < num_layers - 1;
            grad = layer.backward(&grad, use_activation);
        }
    }

    pub fn predict(&mut self, state: &[f32; 4]) -> usize {
        let output = self.forward(state);
        output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    pub fn copy_weights_from(&mut self, other: &Network) {
        for (layer, other_layer) in self.layers.iter_mut().zip(other.layers.iter()) {
            layer.weights = other_layer.weights.clone();
            layer.bias = other_layer.bias.clone();
        }
    }
}

impl Clone for Network {
    fn clone(&self) -> Self {
        Network {
            layers: self.layers.clone(),
        }
    }
}
