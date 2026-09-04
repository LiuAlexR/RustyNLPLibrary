use crate::math::Backend;
use burn::{tensor::activation::softmax, Tensor};

// this is strictly for multinomial regression
// X is size mxf, where m is number of inputs and f is number of features
// y is size mxk, where m is number of inputs and k is number of classes
pub fn logistic_regression(
    X: Tensor<Backend, 2>,
    y: Tensor<Backend, 2>,
    learning_rate: f64,
) -> Tensor<Backend, 2> {
    let d = Default::default();
    let X = add_bias(X);
    let weight_shape = [y.dims()[1], X.dims()[1]];
    let mut weights = Tensor::<Backend, 2>::zeros(weight_shape, &d);
    let m = y.dims()[0] as f64;

    for _ in 0..1000 {
        weights = grad(X.clone(), y.clone(), weights, learning_rate, m);
    }
    weights
}

// adding bias column
fn add_bias(input: Tensor<Backend, 2>) -> Tensor<Backend, 2> {
    let f = input.dims()[0];
    let ones = Tensor::<Backend, 2>::ones([f, 1], &Default::default());
    Tensor::<Backend, 2>::cat(vec![input, ones], 1)
}

fn grad(
    X: Tensor<Backend, 2>,
    y: Tensor<Backend, 2>,
    weights: Tensor<Backend, 2>,
    learning_rate: f64,
    m: f64,
) -> Tensor<Backend, 2> {
    let pred = softmax(X.clone().matmul(weights.clone().transpose()), 1);
    let mut s = pred.sub(y);
    s = s.transpose().div_scalar(m).matmul(X);
    weights.sub(s.mul_scalar(learning_rate))
}
