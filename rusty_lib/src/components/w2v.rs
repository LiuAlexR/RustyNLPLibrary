use std::collections::HashMap;

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
    pub fn embedding_matrix(&self) -> Tensor<Backend, 2> {
        self.target_matrix.clone()
    }

    pub fn train_naive(&mut self, corpus: &[usize], unigram: &[usize], batch_size: usize) {
        let (cum, sum) = build_cumulative(unigram); // computed once, not per sample

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
            self.train_batch(chunk, &cum, sum);
        }
    }

    fn train_batch(&mut self, pairs: &[(usize, usize)], cum: &[usize], sum: usize) {
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
                let mut c_neg = sample_negative(cum, sum);
                while c_neg == target {
                    c_neg = sample_negative(cum, sum);
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

    pub fn embedding(&self, idx: usize) -> Tensor<Backend, 1> {
        get_row(&self.target_matrix, idx)
    }

    pub fn cosine_similarity(&self, a: usize, b: usize) -> f32 {
        let va = get_row(&self.target_matrix, a);
        let vb = get_row(&self.target_matrix, b);
        let dot = (va.clone() * vb.clone()).sum();
        let norm_a = (va.clone() * va).sum().sqrt();
        let norm_b = (vb.clone() * vb).sum().sqrt();
        let cos = dot / (norm_a * norm_b);
        cos.into_scalar()
    }

    pub fn nearest_to_analogy(&self, a: usize, b: usize, c: usize) -> usize {
        // b - a + c ≈ target
        let va = get_row(&self.target_matrix, a);
        let vb = get_row(&self.target_matrix, b);
        let vc = get_row(&self.target_matrix, c);
        let query = vb - va + vc;

        let vocab_size = self.vocabulary.len();
        let mut best_idx = 0;
        let mut best_sim = f32::MIN;
        for idx in 0..vocab_size {
            if idx == a || idx == b || idx == c {
                continue; // conventional to exclude the input words themselves
            }
            let candidate = get_row(&self.target_matrix, idx);
            let dot = (query.clone() * candidate.clone()).sum();
            let norm_q = (query.clone() * query.clone()).sum().sqrt();
            let norm_c = (candidate.clone() * candidate).sum().sqrt();
            let sim: f32 = (dot / (norm_q * norm_c)).into_scalar();
            if sim > best_sim {
                best_sim = sim;
                best_idx = idx;
            }
        }
        best_idx
    }
}

/// Builds the canonical token -> id map from vocabulary position.
pub fn build_vocab_map(vocabulary: &[String]) -> HashMap<String, i64> {
    vocabulary
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i as i64))
        .collect()
}

fn get_row(m: &Tensor<Backend, 2>, idx: usize) -> Tensor<Backend, 1> {
    let dim = m.dims()[1];
    m.clone().slice([idx..idx + 1, 0..dim]).squeeze().detach()
}

/// Builds a cumulative-sum array from unigram counts, once, for fast weighted sampling.
fn build_cumulative(unigram: &[usize]) -> (Vec<usize>, usize) {
    let mut cum = Vec::with_capacity(unigram.len());
    let mut running = 0usize;
    for &w in unigram {
        running += w;
        cum.push(running);
    }
    (cum, running)
}

/// Draws a weighted-random index in O(log vocab_size) via binary search,
/// instead of the old O(vocab_size) linear scan.
fn sample_negative(cum: &[usize], sum: usize) -> usize {
    let target = rand::random_range(0..sum);
    cum.partition_point(|&c| c <= target)
}

pub fn build_model<'a>(
    vocab: &'a [String],
    dimension: usize,
    window_size: usize,
    num_of_negs: usize,
    learning_rate: f64,
) -> (Word2Vec<'a>, HashMap<String, i64>) {
    let vocab_size = vocab.len();
    (
        Word2Vec {
            vocabulary: vocab,
            target_matrix: create_random_matrix(vocab_size, dimension),
            context_matrix: create_random_matrix(vocab_size, dimension),
            window_size,
            k: num_of_negs,
            dim: dimension,
            learning_rate,
        },
        build_vocab_map(vocab),
    )
}
