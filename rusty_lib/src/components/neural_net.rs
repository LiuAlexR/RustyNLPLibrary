// each hidden unit has a weight vector and a bias
// We have weight matrix W to represent all weights of hidden layer
// vector b for biases
//
// Wji represents weight of ith input xi and jth hidden unit hi
//
// Output now is h = activator(Wx + b)
//
// h is a representation of input
// output layer takes h and computes a final output
//
// output layer does z = Uh, where U is a weight matrix
// Uij is weight from unit hj to unit i in the output layer
//
// We then softmax z to get probabilities
//
// bias is now just added at the end of input rather than its own thing
// so now its like h = act(Wx)
//
// with regard to matrices
// H = act(X * tranpose of W + b), where X's row vector is input, so shape [mxd]
// Z = H x tranpose of U
// Y = softmax(Z)
//
// Two ways to take in embeddings as input: pooling and concatenation
//
// Concatenation: Take a shape of [Nxd], where I think N means # of input vectors, D is dimensions
// We reshape it into a [1xdN] vector, so all the vectors presumably are right next to each other
//
// derivative of RELU is {0 for z < 0, 1 otherwise}

use std::collections::HashMap;

use crate::math::Backend;
use burn::Tensor;

// first step, map tokens to embeddings
// second step, concatenate into a [1xnd] array
// third, for all layers that isn't output, create method of multipling weights,
// propagating backwards, yea

// pub fn map_tokens() -> HashMap<String, Tensor<Backend,2> {}
