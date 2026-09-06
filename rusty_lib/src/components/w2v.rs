// tensor only word2vec
use crate::math::{create_random_matrix, Backend};
use burn::tensor::{activation::sigmoid, IndexingUpdateOp, Int, Tensor};

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
    pub fn train_naive(&mut self, corpus: &[usize], unigram: &[usize], batch_size: usize) {
        let unigram_sum: usize = unigram.iter().sum();

        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for i in self.window_size..(corpus.len() - self.window_size) {
            let target = corpus[i];
            for j in (1..=self.window_size).rev() {
                pairs.push((target, corpus[i - j]));
            }
            for j in 0..self.window_size {
                pairs.push((target, corpus[i + j]));
            }
        }

        for chunk in pairs.chunks(batch_size) {
            self.train_batch(chunk, unigram, unigram_sum);
        }
    }
    fn train_batch(&mut self, pairs: &[(usize, usize)], unigram: &[usize], unigram_sum: usize) {
        let device = self.target_matrix.device();
        let n = pairs.len();

        let target_idx: Vec<i64> = pairs.iter().map(|&(t, _)| t as i64).collect();
        let ctx_idx: Vec<i64> = pairs.iter().map(|&(_, c)| c as i64).collect();

        let target_idx_t = Tensor::<Backend, 1, Int>::from_data(target_idx.as_slice(), &device);
        let ctx_idx_t = Tensor::<Backend, 1, Int>::from_data(ctx_idx.as_slice(), &device);
        let target_rows = self.target_matrix.clone().select(0, target_idx_t.clone()); // [n, dim]
        let ctx_rows = self.context_matrix.clone().select(0, ctx_idx_t.clone()); // [n, dim]

        let dots = (target_rows.clone() * ctx_rows.clone()).sum_dim(1); // [n, 1]
        let errors = sigmoid(dots) - 1.0; // [n, 1]

        let mut neg_idx: Vec<i64> = Vec::with_capacity(n * self.k);
        for &(target, _) in pairs {
            for _ in 0..self.k {
                let mut c_neg = get_weighted_index(unigram, unigram_sum);
                while c_neg == target {
                    c_neg = get_weighted_index(unigram, unigram_sum);
                }
                neg_idx.push(c_neg as i64);
            }
        }
        let neg_idx_t = Tensor::<Backend, 1, Int>::from_data(neg_idx.as_slice(), &device);
        let neg_rows = self.context_matrix.clone().select(0, neg_idx_t.clone()); // [n*k, dim]

        // NOTE: repeat_dim must produce [t0,t0,...(k times),t1,t1,...] to align with
        // neg_idx's per-pair-then-per-k ordering above
        // verify against Burn's actual repeat_dim semantics before trusting this
        let target_rows_rep = target_rows.clone().repeat_dim(0, self.k); // [n*k, dim]
        let neg_dots = (target_rows_rep.clone() * neg_rows.clone()).sum_dim(1); // [n*k, 1]
        let neg_delta = sigmoid(neg_dots);

        let updated_neg_rows = neg_rows.clone()
            - target_rows_rep
                .clone()
                .mul(neg_delta.clone())
                .mul_scalar(self.learning_rate);
        let delta_neg = updated_neg_rows - neg_rows; // delta = new - old
        self.context_matrix = self
            .context_matrix
            .clone()
            .select_assign(0, neg_idx_t, delta_neg, IndexingUpdateOp::Add)
            .detach();

        let updated_ctx = ctx_rows.clone()
            - target_rows
                .clone()
                .mul(errors.clone())
                .mul_scalar(self.learning_rate);
        let delta_ctx = updated_ctx - ctx_rows.clone();
        self.context_matrix = self
            .context_matrix
            .clone()
            .select_assign(0, ctx_idx_t, delta_ctx, IndexingUpdateOp::Add)
            .detach();

        let positive_gradient = ctx_rows.mul(errors);
        let delta_target = positive_gradient.mul_scalar(-self.learning_rate);
        self.target_matrix = self
            .target_matrix
            .clone()
            .select_assign(0, target_idx_t, delta_target, IndexingUpdateOp::Add)
            .detach();
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
    m.clone().slice([idx..idx + 1, 0..dim]).squeeze().detach()
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
