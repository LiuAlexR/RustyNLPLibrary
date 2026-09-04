use std::collections::HashMap;

use crate::math::{create_random_matrix, Backend};
use burn::{
    tensor::{activation::relu, Int},
    Tensor,
};

const LEARNING_RATE: f64 = 3.4;
const NUM_LAYERS: i64 = 3;
const NUM_HIDDEN_NODES: i64 = 10;

// X is going to be past n words
// y is going to be the one hot vector of n
// pass in weights as well?

pub fn use_relu(weights: Tensor<Backend, 2>) -> Tensor<Backend, 2> {
    relu(weights)
}
pub fn train<Activator>(
    input: &[String],
    embedding_matrix: Tensor<Backend, 2>,
    token_map: &HashMap<String, i64>,
    act: Activator,
) -> Tensor<Backend, 2>
where
    Activator: Fn(Tensor<Backend, 2>) -> Tensor<Backend, 2>,
{
    let concatenated = concatenate_embeddings(input, token_map, embedding_matrix);
    concatenated
}

fn one_pass<Activator>(
    X: Tensor<Backend, 2>,
    y: Tensor<Backend, 1>,
    weights: Tensor<Backend, 2>,
    act: Activator,
) -> Tensor<Backend, 2>
where
    Activator: Fn(Tensor<Backend, 2>) -> Tensor<Backend, 2>,
{
    create_random_matrix()
}

fn concatenate_embeddings(
    input: &[String],
    token_map: &HashMap<String, i64>,
    embedding_matrix: Tensor<Backend, 2>,
) -> Tensor<Backend, 2> {
    let ids: Vec<i64> = input.iter().map(|w| token_map[w]).collect();
    let idx = Tensor::<Backend, 1, Int>::from_data(ids.as_slice(), &Default::default());
    embedding_matrix.select(0, idx)
}
