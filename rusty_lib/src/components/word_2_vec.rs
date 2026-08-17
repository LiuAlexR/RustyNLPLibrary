use burn::Tensor;
use crate::math::{Backend, VOCAB};
pub struct word_2_vec<'a> {
    vocabulary: &'a [usize],
    target_matrix: Tensor<Backend, 2>,
    context_matrix: Tensor<Backend, 2>,
    target_slice: Vec<Vec<f64>>,
    context_slice: Vec<Vec<f64>>,
    window_size: usize, // Window size
    k: usize, // Number of negative contexts
    dim: usize, // Dimension of the embeddings
    learning_rate: f64, // Learning rate of algorithm
}
impl<'a> word_2_vec<'a> {
    pub fn init(&mut self, vocab: &'a [usize]) -> () {
        self.vocabulary = vocab;
        let vocab_size = vocab.len();
        self.target_matrix = crate::math::create_random_matrix_custom_dimensions(vocab_size, self.dim);
        self.context_matrix = crate::math::create_random_matrix_custom_dimensions(vocab_size, self.dim);
    }
    pub fn init_naive(&mut self, vocab: &'a [usize]) -> () {
        self.vocabulary = vocab;
        let vocab_size = vocab.len();
        self.target_matrix = crate::math::create_random_matrix_custom_dimensions(vocab_size, self.dim);
        self.context_matrix = crate::math::create_random_matrix_custom_dimensions(vocab_size, self.dim);
    }
    pub fn train_naive(&mut self, corpus: &'a [usize], unigram: &'a [usize]) -> () {
        let unigram_sum: usize =  unigram.iter().sum();
        for i in self.window_size..(corpus.len() - self.window_size) {
            let target = corpus[i];
            // k below target
            for j in (1..=(self.window_size)).rev() {
                let c_pos = corpus[i - j];
                let mut negative_gradient: f64 = 0f64;
                for _ in 0..self.k {
                    let mut c_neg = get_weighted_index(unigram, unigram_sum);
                    while c_neg == target {
                        c_neg = get_weighted_index(unigram, unigram_sum);
                    }
                    // Adjust c_neg weight
                    
                    // Add to negative gradient
                }
                // Adjust target, c_pos weights
            }
        }
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