use std::collections::HashMap;

use crate::{
    components::neural_net::{forward_pass, use_gelu, use_relu},
    math::{create_random_matrix, create_random_vector, Backend},
};
use burn::{
    tensor::{activation::softmax, Bool, Int, TensorData},
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
    gamma_pre: Tensor<Backend, 1>,  // [1xd]
    beta_pre: Tensor<Backend, 1>,   // [1xd]
    gamma_post: Tensor<Backend, 1>, // [1xd]
    beta_post: Tensor<Backend, 1>,  // [1xd]
    wq: Tensor<Backend, 2>,         // [dxd]
    wk: Tensor<Backend, 2>,         // [dxd]
    wv: Tensor<Backend, 2>,         // [dxd]
    wo: Tensor<Backend, 2>,         // [dxd]
    w_ffn_h: Tensor<Backend, 2>,    // [dxff]
    w_ffn_o: Tensor<Backend, 2>,    // [ffxd]
}

pub fn run_block(X: Tensor<Backend, 3>, b: &Block, d: usize, heads: usize) -> Tensor<Backend, 3> {
    let l = layer_norm(X.clone(), b.gamma_pre.clone(), b.beta_pre.clone());
    let A = X.clone().add(calculate_attention(l, heads, d, b));

    let l = layer_norm(A.clone(), b.gamma_post.clone(), b.beta_post.clone());

    let F = A.clone().add(forward_pass(
        l,
        b.w_ffn_h.clone(),
        b.w_ffn_o.clone(),
        use_gelu,
    ));
    F
}

/// input shape : batch x len x dimension
pub fn calculate_attention(
    X: Tensor<Backend, 3>,
    num_heads: usize,
    d: usize,
    b: &Block,
) -> Tensor<Backend, 3> {
    // shapes are still batch x len x d
    let Q = X.clone().matmul(b.wq.clone().unsqueeze());
    let K = X.clone().matmul(b.wk.clone().unsqueeze());
    let V = X.clone().matmul(b.wv.clone().unsqueeze());

    let batch_size = Q.dims()[0];
    let len = Q.dims()[1];

    // shapes are now batch x len x num_heads x (d / num_heads)
    let Q: Tensor<Backend, 4> = Q.reshape([batch_size, len, num_heads, d / num_heads]);
    let K: Tensor<Backend, 4> = K.reshape([batch_size, len, num_heads, d / num_heads]);
    let V: Tensor<Backend, 4> = V.reshape([batch_size, len, num_heads, d / num_heads]);

    // shapes are now batch x num_heads x len x (d / num_heads)
    let Q = Q.swap_dims(1, 2);
    let K = K.swap_dims(1, 2);
    let V = V.swap_dims(1, 2);

    // transposing K along the last 2 dims
    // K is now of shape batch x num_heads (d / num_heads) x len
    let K = K.swap_dims(2, 3);

    // product shape is now batch x heads x len x len
    let p = Q.matmul(K).div_scalar((d as f64).sqrt());
    let mask = Tensor::<Backend, 4, Bool>::tril_mask(p.dims(), 0, &Default::default());
    let p = p.mask_fill(mask, f64::MIN);

    // softmaxing over the last dim, which represents the keys
    let p = softmax(p, 3);

    // shape is now batch x heads x len x d_h
    let p = p.matmul(V);

    // transposing back
    // shape is now batch x len x heads x d_h
    let p = p.swap_dims(1, 2);

    // reshaping into shape batch x len x d
    let p = p.reshape([batch_size, len, d]);

    p.matmul(b.wo.clone().unsqueeze())
}

/// Applies LayerNorm to input
pub fn layer_norm(
    X: Tensor<Backend, 3>,
    gamma: Tensor<Backend, 1>,
    beta: Tensor<Backend, 1>,
) -> Tensor<Backend, 3> {
    let mean = X.clone().mean_dim(2);
    let variance = X.clone().sub(mean.clone()).powf_scalar(2.).mean_dim(2);

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
            gamma_pre: create_random_vector(d),
            beta_pre: create_random_vector(d),
            gamma_post: create_random_vector(d),
            beta_post: create_random_vector(d),
            wq: create_random_matrix(d, d),
            wk: create_random_matrix(d, d),
            wv: create_random_matrix(d, d),
            wo: create_random_matrix(d, d),
            w_ffn_h: create_random_matrix(d, ff),
            w_ffn_o: create_random_matrix(ff, d),
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
