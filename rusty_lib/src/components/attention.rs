// To calculate Q,K,V matrices
// we take X, which is the input matrix of size nxd, where n is number of tokens
// and d is number of dimensions in the embedding
// and do Q = XWq, K = XWk, V = XWv
//
// Attention score = Q x Transpose of K
// Scale it down by dividing it by sqrt(dk), dk is dimension of key vectors
// Attention weights = softmax(the above)
// Output = Attention weights x V
//
// For multihead attention, we split input sequence into smaller segemnets,
// process each separately, then concatenate all the weight matrices together
//
// Afterwards, we process through FFN
// FFN(x) = ReLU(xW1 + b1)W2 + b2
// where x is the input of the activation function
// W1 and W2 are matrices, and b1 and b2 are bias vectors
//

use std::collections::HashMap;

use crate::math::{create_random_matrix, Backend};
use burn::{
    tensor::{Int, TensorData},
    Tensor,
};

const DK: i64 = 64;
const DV: i64 = 64;
const D: i64 = 512;
const BLOCKS: i64 = 2;
const HEADS: i64 = 4;
const EPSILON: f64 = 1e-8;

// Implement method to create final input matrix of [Nxd]
// Then method to calculate attention
//  make sure to save weights in a vector or somewhere,
//  remember, Q,K,V weights for each head, in each block
//  If there are 5 heads and 4 blocks, then there are a total of
//  3 x 5 x 4 = 60 weight matrices
// Implement LayerNorm

struct Block {
    q: Tensor<Backend, 2>,     // [dxd]
    k: Tensor<Backend, 2>,     // [dxd]
    v: Tensor<Backend, 2>,     // [dxd]
    o: Tensor<Backend, 2>,     // [dxd]
    ffn_h: Tensor<Backend, 2>, // [dxff]
    ffn_o: Tensor<Backend, 2>, // [ffxd]
}

pub fn forward_pass() {}

/// Applies LayerNorm to input
pub fn layer_norm(
    X: Tensor<Backend, 2>,
    gamma: Tensor<Backend, 1>,
    beta: Tensor<Backend, 1>,
) -> Tensor<Backend, 2> {
    let mean = X.clone().mean_dim(1);
    let variance = X.clone().sub(mean.clone()).powf_scalar(2.).mean_dim(1);

    let normalized = X.sub(mean).div(variance.add_scalar(EPSILON).sqrt());

    normalized.mul(gamma.unsqueeze()).add(beta.unsqueeze())
}

/// initializes weights of all heads in all blocks
///
/// ith block in vec is indicative of a block
pub fn init_weights(blocks: i64, d: i64, ff: i64) -> Vec<Block> {
    let mut res: Vec<Block> = Vec::with_capacity(blocks as usize);

    for _ in 0..blocks {
        res.push(Block {
            q: create_random_matrix(d, d),
            k: create_random_matrix(d, d),
            v: create_random_matrix(d, d),
            o: create_random_matrix(d, d),
            ffn_h: create_random_matrix(d, ff),
            ffn_o: create_random_matrix(ff, d),
        });
    }
    res
}

/// Creates combined embeddings from word and positional embeddings
///
/// Returns a 2D Tensor of shape [Nxd] where `N` is number of tokens and
/// `d` is modal dimensionality
pub fn generate_combined_embeddings(
    tokens: &[String],
    map: &HashMap<String, i64>,
    embedding_matrix: Tensor<Backend, 2>,
    d: usize,
) -> Tensor<Backend, 2> {
    let positional_embeddings = generate_positional_embeddings(tokens.len(), d);

    let ids: Vec<i64> = tokens.iter().map(|t| map[t]).collect();

    let device = embedding_matrix.device();
    let indices =
        Tensor::<Backend, 1, Int>::from_data(TensorData::new(ids, [tokens.len()]), &device);

    let words = embedding_matrix.select(0, indices);
    words.add(positional_embeddings)
}

/// Generates sinusoidal position embeddings for `length` tokens of dimensionality `d`
///
/// Returns a 2D Tensor where each row vector `i` is the positional embedding for the `ith` token
fn generate_positional_embeddings(length: usize, d: usize) -> Tensor<Backend, 2> {
    let mut res: Vec<Vec<f64>> = vec![vec![]; length];

    for pos in 0..length {
        let pos = pos as f64;
        let mut v: Vec<f64> = vec![0.; d];

        for i in 0..d / 2 {
            let denominator = 10000_f64.powf(2. * i as f64 / d as f64);
            v[2 * i] = (pos / denominator).sin();
            v[2 * i + 1] = (pos / denominator).cos();
        }
        res[pos as usize] = v.clone();
    }

    let shape = [length, d];
    let t = TensorData::new(res.into_iter().flatten().collect(), shape);
    Tensor::<Backend, 2>::from_data(t, &Default::default())
}
