// This neural net implementation is only going to handle 2 layers
// Since the Feedforward layer for a Transformer block is only 2 layers

use crate::{
    components::logistic_regression::add_bias,
    math::{create_random_matrix, Backend},
};
use burn::{
    tensor::{
        activation::{relu, softmax},
        Int,
    },
    Tensor,
};
use std::collections::HashMap;

const LEARNING_RATE: f64 = 3.4;
const NUM_HIDDEN_NODES: i64 = 3;
const EPOCHS: i64 = 2;

type Activator = fn(Tensor<Backend, 2>) -> Tensor<Backend, 2>;

pub fn use_relu(weights: Tensor<Backend, 2>) -> Tensor<Backend, 2> {
    relu(weights)
}

/// Takes X,y, optional previous weights, vocab_size, and activator function
/// to run a neural network on
///
/// X - Tensor of shape [1xnd] where n is number of inputs and d is embedding_dimensions;
///     represents concatenated embedding
/// y - one-hot vector of the correct token that comes after the sequence
/// weights - optional W,U weights to pass in
/// vocab_size - how many tokens in memory
/// act - activator function
pub fn train(
    X: Tensor<Backend, 2>,
    y: Tensor<Backend, 1>,
    weights: Option<Vec<Tensor<Backend, 2>>>,
    vocab_size: i64,
    act: Activator,
) -> (Tensor<Backend, 2>, Tensor<Backend, 2>) {
    let X = add_bias(X);

    let (mut W, mut U) = match weights {
        Some(w) => (w[0].clone(), w[1].clone()),
        None => (
            create_random_matrix(X.dims()[1] as i64, NUM_HIDDEN_NODES).require_grad(),
            create_random_matrix(NUM_HIDDEN_NODES, vocab_size).require_grad(),
        ),
    };

    for _ in 0..EPOCHS {
        (W, U) = one_pass(X.clone(), y.clone(), W, U, act);
    }
    (W, U)
}

/// Executes one forward and backward pass
///
/// X - concatenated embedding of size [1xnd]
/// y - one-hot vector of correct token to predict
/// W - weight matrix for hidden layer
/// U - weight matrix for output layer
/// act - activator function
pub fn one_pass(
    X: Tensor<Backend, 2>,
    y: Tensor<Backend, 1>,
    W: Tensor<Backend, 2>,
    U: Tensor<Backend, 2>,
    act: Activator,
) -> (Tensor<Backend, 2>, Tensor<Backend, 2>) {
    let output = forward_pass(X.clone(), W.clone(), U.clone(), act);
    let sm = softmax(output, 1);
    // cross entropy loss
    let loss = sm.log().mul(y.unsqueeze()).sum().neg();
    backward_pass(W, U, loss.unsqueeze())
}

/// Outputs result of forward pass
///
/// X - concatenated embedding of size [1xnd]
/// W - weight matrix for hidden layer
/// U - weight matrix for output layer
/// act - activator function
pub fn forward_pass(
    X: Tensor<Backend, 2>,
    W: Tensor<Backend, 2>,
    U: Tensor<Backend, 2>,
    act: Activator,
) -> Tensor<Backend, 2> {
    act(X.matmul(W)).matmul(U)
}

/// outputs updated weights
/// backward_pass(W,U) -> (W,U)
///
/// W - weight matrix for hidden layer
/// U - weight matrix for output layer
/// output - loss of output of forward pass
pub fn backward_pass(
    W: Tensor<Backend, 2>,
    U: Tensor<Backend, 2>,
    output: Tensor<Backend, 2>,
) -> (Tensor<Backend, 2>, Tensor<Backend, 2>) {
    let grads = output.backward();
    let w_grad = W.grad(&grads).unwrap();
    let u_grad = U.grad(&grads).unwrap();

    let W_new = Tensor::from_inner(W.inner().sub(w_grad.mul_scalar(LEARNING_RATE))).require_grad();
    let U_new = Tensor::from_inner(U.inner().sub(u_grad.mul_scalar(LEARNING_RATE))).require_grad();

    (W_new, U_new)
}

/// takes an array of tokens and outputs a [1xnd] embedding
///
/// input - array of tokens
/// token_map - mapping of strings to their indices
/// embedding_matrix - |V|xd matrix holding all embeddings
pub fn concatenate_embeddings(
    input: &[String],
    token_map: &HashMap<String, i64>,
    embedding_matrix: Tensor<Backend, 2>,
) -> Tensor<Backend, 2> {
    let ids: Vec<i64> = input.iter().map(|w| token_map[w]).collect();
    let idx = Tensor::<Backend, 1, Int>::from_data(ids.as_slice(), &Default::default());
    embedding_matrix.select(0, idx)
}
