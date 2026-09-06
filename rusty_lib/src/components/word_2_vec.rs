use std::ops::Div;

use crate::math::{create_random_matrix, Backend, DIMENSIONS, VOCAB};
use burn::Tensor;
use rand::distr::{Distribution, Uniform};
/// This is Word2Vec.
pub struct Word2Vec<'a> {
    vocabulary: &'a [String],
    target_matrix: Tensor<Backend, 2>,
    context_matrix: Tensor<Backend, 2>,
    target_slice: Vec<Vec<f64>>,
    context_slice: Vec<Vec<f64>>,
    window_size: usize, // Window size
    k: usize,           // Number of negative contexts
    dim: usize,         // Dimension of the embeddings
    learning_rate: f64, // Learning rate of algorithm
}
impl<'a> Word2Vec<'a> {
    pub fn init(&mut self, vocab: &'a [String]) -> () {
        self.vocabulary = vocab;
        let vocab_size = vocab.len();
        self.target_matrix =
            crate::math::create_random_matrix_custom_dimensions(vocab_size, self.dim);
        self.context_matrix =
            crate::math::create_random_matrix_custom_dimensions(vocab_size, self.dim);
    }
    // pub fn init_naive(&mut self, vocab: &'a [String]) -> () {
    //     let range = Uniform::new(-1.0, 1.0).expect("Failed");
    //     let mut rng = rand::rng();
    //     self.vocabulary = vocab;
    //     let vocab_size = vocab.len();
    //     for _ in 0..vocab_size {}
    //     self.target_slice = vec![(0..self.dim).map(|_| range.sample(&mut rng)).collect(); vocab_size];
    //     self.target_slice = vec![(0..self.dim).map(|_| range.sample(&mut rng)).collect(); vocab_size];
    // }
    pub fn train_naive(&mut self, corpus: &'a [usize], unigram: &'a [usize]) -> () {
        let unigram_sum: usize = unigram.iter().sum();
        for i in self.window_size..(corpus.len() - self.window_size) {
            let target = corpus[i];
            // k below target
            for j in (1..=(self.window_size)).rev() {
                let c_pos = corpus[i - j];
                let mut negative_gradient: Vec<f64> = vec![0.0; self.dim];
                for _ in 0..self.k {
                    let mut c_neg = get_weighted_index(unigram, unigram_sum);
                    while c_neg == target {
                        c_neg = get_weighted_index(unigram, unigram_sum);
                    }
                    // Adjust c_neg weight
                    let delta_c = sigmoid(dot_product(
                        &self.context_slice[c_neg],
                        &self.target_slice[target],
                    ));
                    let new_c_neg = vector_diff(
                        &self.context_slice[c_neg],
                        &vector_x_constant(
                            &self.target_slice[target],
                            self.learning_rate * delta_c,
                        ),
                    );
                    self.context_slice[c_neg] = new_c_neg;

                    // Add to negative gradient
                    negative_gradient = vector_sum(
                        &negative_gradient,
                        &vector_x_constant(&self.context_slice[c_neg], delta_c),
                    );
                }
                // Adjust target, c_pos weights
                let dot = dot_product(&self.context_slice[c_pos], &self.target_slice[target]);
                let error = sigmoid(dot) - 1.0;

                let positive_gradient = vector_x_constant(&self.context_slice[c_pos], error);

                self.context_slice[c_pos] = vector_diff(
                    &self.context_slice[c_pos],
                    &vector_x_constant(&self.target_slice[target], self.learning_rate * error),
                );

                let target_gradient = vector_sum(&positive_gradient, &negative_gradient);

                self.target_slice[target] = vector_diff(
                    &self.target_slice[target],
                    &vector_x_constant(&target_gradient, self.learning_rate),
                );
            }
            for j in 0..(self.window_size) {
                let c_pos = corpus[i + j];
                let mut negative_gradient: Vec<f64> = vec![0.0; self.dim];
                for _ in 0..self.k {
                    let mut c_neg = get_weighted_index(unigram, unigram_sum);
                    while c_neg == target {
                        c_neg = get_weighted_index(unigram, unigram_sum);
                    }
                    // Adjust c_neg weight
                    let delta_c = sigmoid(dot_product(
                        &self.context_slice[c_neg],
                        &self.target_slice[target],
                    ));
                    let new_c_neg = vector_diff(
                        &self.context_slice[c_neg],
                        &vector_x_constant(
                            &self.target_slice[target],
                            self.learning_rate * delta_c,
                        ),
                    );
                    self.context_slice[c_neg] = new_c_neg;

                    // Add to negative gradient
                    negative_gradient = vector_sum(
                        &negative_gradient,
                        &vector_x_constant(&self.context_slice[c_neg], delta_c),
                    );
                }
                let dot = dot_product(&self.context_slice[c_pos], &self.target_slice[target]);
                let error = sigmoid(dot) - 1.0;

                let positive_gradient = vector_x_constant(&self.context_slice[c_pos], error);

                self.context_slice[c_pos] = vector_diff(
                    &self.context_slice[c_pos],
                    &vector_x_constant(&self.target_slice[target], self.learning_rate * error),
                );

                let target_gradient = vector_sum(&positive_gradient, &negative_gradient);

                self.target_slice[target] = vector_diff(
                    &self.target_slice[target],
                    &vector_x_constant(&target_gradient, self.learning_rate),
                );
            }
        }
    }
    pub fn print_vec(&self, index: usize) -> () {
        println!("{:?}", self.target_slice[index]);
    }
    pub fn adjust_training_rate(&mut self, new_rate: f64) -> () {
        self.learning_rate = new_rate;
    }
}
fn get_weighted_index(unigram: &[usize], sum: usize) -> usize {
    let mut target = rand::random_range(0..sum);

    for (index, &weight) in unigram.iter().enumerate() {
        if target < weight {
            return index;
        }
        target -= weight;
    }
    unigram.len().saturating_sub(1)
}
fn dot_product(v1: &[f64], v2: &[f64]) -> f64 {
    if v1.len() != v2.len() {
        panic!("Error. Tries to take the dot product of vectors with different sizes");
    }
    let mut sum: f64 = 0f64;
    for i in 0..v1.len() {
        sum = sum + v1[i] * v2[i];
    }
    sum
}
fn vector_diff(v1: &[f64], v2: &[f64]) -> Vec<f64> {
    if v1.len() != v2.len() {
        panic!("Error. Tries to take the dot product of vectors with different sizes");
    }
    let mut res: Vec<f64> = Vec::new();
    for i in 0..v1.len() {
        res.push(v1[i] - v2[i]);
    }
    res
}
fn vector_sum(v1: &[f64], v2: &[f64]) -> Vec<f64> {
    if v1.len() != v2.len() {
        panic!("Error. Tries to take the dot product of vectors with different sizes");
    }
    let mut res: Vec<f64> = Vec::new();
    for i in 0..v1.len() {
        res.push(v1[i] + v2[i]);
    }
    res
}
fn vector_x_constant(v1: &[f64], c: f64) -> Vec<f64> {
    let mut res: Vec<f64> = Vec::new();
    for i in 0..v1.len() {
        res.push(v1[i] * c);
    }
    res
}
fn sigmoid_vec(v: &[f64]) -> Vec<f64> {
    let mut res: Vec<f64> = Vec::new();
    for i in v {
        let x = -1.0 * *i;
        let ex = x.exp();
        let sig = 1.0f64.div(1.0 + ex);
        res.push(sig);
    }
    res
}
fn sigmoid(v: f64) -> f64 {
    let x = -1.0 * v;
    let ex = x.exp();
    let sig = 1.0f64.div(1.0 + ex);
    sig
}
pub fn build_model<'a>(
    vocab: &'a [String],
    dimension: usize,
    window_size_: usize,
    num_of_negs: usize,
    learning_rate_: f64,
) -> Word2Vec<'a> {
    let range = Uniform::new(-1.0, 1.0).expect("Failed");
    let mut rng = rand::rng();
    let vocab_size = vocab.len();
    let mut rand_vec_1: Vec<Vec<f64>> = Vec::new();
    let mut rand_vec_2: Vec<Vec<f64>> = Vec::new();
    for _ in 0..vocab_size {
        rand_vec_1.push((0..dimension).map(|_| range.sample(&mut rng)).collect());
        rand_vec_2.push((0..dimension).map(|_| range.sample(&mut rng)).collect());
    }
    Word2Vec {
        vocabulary: vocab,
        target_matrix: create_random_matrix(VOCAB as i64, DIMENSIONS as i64),
        context_matrix: create_random_matrix(VOCAB as i64, DIMENSIONS as i64),
        target_slice: rand_vec_1,
        context_slice: rand_vec_2,
        window_size: window_size_,
        k: num_of_negs,
        dim: dimension,
        learning_rate: learning_rate_,
    }
}
