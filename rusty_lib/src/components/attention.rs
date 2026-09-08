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

use crate::math::Backend;
use burn::{
    tensor::{Float, TensorData},
    Tensor,
};

const DK: i64 = 64;
const DV: i64 = 64;
const D: i64 = 512;

// Implement method to create final input matrix of [Nxd]
// Then method to calculate attention
//  make sure to save weights in a vector or somewhere,
//  remember, Q,K,V weights for each head, in each block
//  If there are 5 heads and 4 blocks, then there are a total of
//  3 x 5 x 4 = 60 weight matrices
// Implement LayerNorm

/// Generates sinusoidal position embeddings for `length` tokens of dimensionality `d`
///
/// Returns a 2D Tensor where each row vector `i` is the positional embedding for the `ith` token
pub fn generate_positional_embeddings(length: usize, d: usize) -> Tensor<Backend, 2> {
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
