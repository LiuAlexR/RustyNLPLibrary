// tensor only word2vec
// needs review!

use crate::math::{create_random_matrix, Backend};
use burn::tensor::{activation::sigmoid, Tensor};

pub struct Word2Vec<'a> {
    vocabulary: &'a [String],
    target_matrix: Tensor<Backend, 2>,
    context_matrix: Tensor<Backend, 2>,
    window_size: usize,
    k: usize,
    dim: usize,
    learning_rate: f64,
}

impl<'a> Word2Vec<'a> {
    pub fn train_naive(&mut self, corpus: &[usize], unigram: &[usize]) {
        let unigram_sum: usize = unigram.iter().sum();
        for i in self.window_size..(corpus.len() - self.window_size) {
            let target = corpus[i];
            for j in (1..=self.window_size).rev() {
                self.train_pair(target, corpus[i - j], unigram, unigram_sum);
            }
            for j in 0..self.window_size {
                self.train_pair(target, corpus[i + j], unigram, unigram_sum);
            }
        }
    }

    fn train_pair(&mut self, target: usize, c_pos: usize, unigram: &[usize], unigram_sum: usize) {
        let device = self.target_matrix.device();
        let target_row = get_row(&self.target_matrix, target);
        let mut negative_gradient = Tensor::<Backend, 1>::zeros([self.dim], &device);

        for _ in 0..self.k {
            let mut c_neg = get_weighted_index(unigram, unigram_sum);
            while c_neg == target {
                c_neg = get_weighted_index(unigram, unigram_sum);
            }
            let c_neg_row = get_row(&self.context_matrix, c_neg);
            let delta_c: f32 = sigmoid(target_row.clone().dot(c_neg_row.clone())).into_scalar();

            let updated_c_neg = c_neg_row
                - target_row
                    .clone()
                    .mul_scalar(self.learning_rate * delta_c as f64);
            self.context_matrix =
                set_row(self.context_matrix.clone(), c_neg, updated_c_neg.clone());
            negative_gradient = negative_gradient + updated_c_neg.mul_scalar(delta_c as f64);
        }

        let c_pos_row = get_row(&self.context_matrix, c_pos);
        let error: f32 = sigmoid(target_row.clone().dot(c_pos_row.clone())).into_scalar() - 1.0;

        let positive_gradient = c_pos_row.clone().mul_scalar(error as f64);
        let updated_c_pos = c_pos_row
            - target_row
                .clone()
                .mul_scalar(self.learning_rate * error as f64);
        self.context_matrix = set_row(self.context_matrix.clone(), c_pos, updated_c_pos);

        let target_gradient = positive_gradient + negative_gradient;
        let updated_target = target_row - target_gradient.mul_scalar(self.learning_rate);
        self.target_matrix = set_row(self.target_matrix.clone(), target, updated_target);
    }

    pub fn print_vec(&self, index: usize) {
        println!("{}", get_row(&self.target_matrix, index));
    }

    pub fn adjust_training_rate(&mut self, new_rate: f64) {
        self.learning_rate = new_rate;
    }
}

fn get_row(m: &Tensor<Backend, 2>, idx: usize) -> Tensor<Backend, 1> {
    let dim = m.dims()[1];
    m.clone().slice([idx..idx + 1, 0..dim]).squeeze()
}

fn set_row(m: Tensor<Backend, 2>, idx: usize, new_row: Tensor<Backend, 1>) -> Tensor<Backend, 2> {
    let dim = m.dims()[1];
    let row2d: Tensor<Backend, 2> = new_row.unsqueeze();
    m.slice_assign([idx..idx + 1, 0..dim], row2d)
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

pub fn build_model<'a>(
    vocab: &'a [String],
    dimension: usize,
    window_size: usize,
    num_of_negs: usize,
    learning_rate: f64,
) -> Word2Vec<'a> {
    let vocab_size = vocab.len();
    Word2Vec {
        vocabulary: vocab,
        target_matrix: create_random_matrix(vocab_size as i64, dimension as i64),
        context_matrix: create_random_matrix(vocab_size as i64, dimension as i64),
        window_size,
        k: num_of_negs,
        dim: dimension,
        learning_rate,
    }
}
