use crate::math::Backend;
use burn::{tensor::activation::sigmoid, Tensor};

// Iris dataset (Fisher, 1936) — setosa (label 0) vs versicolor (label 1)
// Columns: sepal_length, sepal_width, petal_length, petal_width (all cm)
pub fn iris_setosa_versicolor() -> (Tensor<Backend, 2>, Tensor<Backend, 2>) {
    let device = Default::default();
    let features_data: Vec<f32> = vec![
        // setosa (label 0), 50 rows
        5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 4.7, 3.2, 1.3, 0.2, 4.6, 3.1, 1.5, 0.2, 5.0, 3.6,
        1.4, 0.2, 5.4, 3.9, 1.7, 0.4, 4.6, 3.4, 1.4, 0.3, 5.0, 3.4, 1.5, 0.2, 4.4, 2.9, 1.4, 0.2,
        4.9, 3.1, 1.5, 0.1, 5.4, 3.7, 1.5, 0.2, 4.8, 3.4, 1.6, 0.2, 4.8, 3.0, 1.4, 0.1, 4.3, 3.0,
        1.1, 0.1, 5.8, 4.0, 1.2, 0.2, 5.7, 4.4, 1.5, 0.4, 5.4, 3.9, 1.3, 0.4, 5.1, 3.5, 1.4, 0.3,
        5.7, 3.8, 1.7, 0.3, 5.1, 3.8, 1.5, 0.3, 5.4, 3.4, 1.7, 0.2, 5.1, 3.7, 1.5, 0.4, 4.6, 3.6,
        1.0, 0.2, 5.1, 3.3, 1.7, 0.5, 4.8, 3.4, 1.9, 0.2, 5.0, 3.0, 1.6, 0.2, 5.0, 3.4, 1.6, 0.4,
        5.2, 3.5, 1.5, 0.2, 5.2, 3.4, 1.4, 0.2, 4.7, 3.2, 1.6, 0.2, 4.8, 3.1, 1.6, 0.2, 5.4, 3.4,
        1.5, 0.4, 5.2, 4.1, 1.5, 0.1, 5.5, 4.2, 1.4, 0.2, 4.9, 3.1, 1.5, 0.2, 5.0, 3.2, 1.2, 0.2,
        5.5, 3.5, 1.3, 0.2, 4.9, 3.6, 1.4, 0.1, 4.4, 3.0, 1.3, 0.2, 5.1, 3.4, 1.5, 0.2, 5.0, 3.5,
        1.3, 0.3, 4.5, 2.3, 1.3, 0.3, 4.4, 3.2, 1.3, 0.2, 5.0, 3.5, 1.6, 0.6, 5.1, 3.8, 1.9, 0.4,
        4.8, 3.0, 1.4, 0.3, 5.1, 3.8, 1.6, 0.2, 4.6, 3.2, 1.4, 0.2, 5.3, 3.7, 1.5, 0.2, 5.0, 3.3,
        1.4, 0.2, // versicolor (label 1), 50 rows
        7.0, 3.2, 4.7, 1.4, 6.4, 3.2, 4.5, 1.5, 6.9, 3.1, 4.9, 1.5, 5.5, 2.3, 4.0, 1.3, 6.5, 2.8,
        4.6, 1.5, 5.7, 2.8, 4.5, 1.3, 6.3, 3.3, 4.7, 1.6, 4.9, 2.4, 3.3, 1.0, 6.6, 2.9, 4.6, 1.3,
        5.2, 2.7, 3.9, 1.4, 5.0, 2.0, 3.5, 1.0, 5.9, 3.0, 4.2, 1.5, 6.0, 2.2, 4.0, 1.0, 6.1, 2.9,
        4.7, 1.4, 5.6, 2.9, 3.6, 1.3, 6.7, 3.1, 4.4, 1.4, 5.6, 3.0, 4.5, 1.5, 5.8, 2.7, 4.1, 1.0,
        6.2, 2.2, 4.5, 1.5, 5.6, 2.5, 3.9, 1.1, 5.9, 3.2, 4.8, 1.8, 6.1, 2.8, 4.0, 1.3, 6.3, 2.5,
        4.9, 1.5, 6.1, 2.8, 4.7, 1.2, 6.4, 2.9, 4.3, 1.3, 6.6, 3.0, 4.4, 1.4, 6.8, 2.8, 4.8, 1.4,
        6.7, 3.0, 5.0, 1.7, 6.0, 2.9, 4.5, 1.5, 5.7, 2.6, 3.5, 1.0, 5.5, 2.4, 3.8, 1.1, 5.5, 2.4,
        3.7, 1.0, 5.8, 2.7, 3.9, 1.2, 6.0, 2.7, 5.1, 1.6, 5.4, 3.0, 4.5, 1.5, 6.0, 3.4, 4.5, 1.6,
        6.7, 3.1, 4.7, 1.5, 6.3, 2.3, 4.4, 1.3, 5.6, 3.0, 4.1, 1.3, 5.5, 2.5, 4.0, 1.3, 5.5, 2.6,
        4.4, 1.2, 6.1, 3.0, 4.6, 1.4, 5.8, 2.6, 4.0, 1.2, 5.0, 2.3, 3.3, 1.0, 5.6, 2.7, 4.2, 1.3,
        5.7, 3.0, 4.2, 1.2, 5.7, 2.9, 4.2, 1.3, 6.2, 2.9, 4.3, 1.3, 5.1, 2.5, 3.0, 1.1, 5.7, 2.8,
        4.1, 1.3,
    ];

    let y_data: Vec<f32> = [vec![0.0; 50], vec![1.0; 50]].concat();

    let features =
        Tensor::<Backend, 1>::from_floats(features_data.as_slice(), &device).reshape([100, 4]);
    let y = Tensor::<Backend, 1>::from_floats(y_data.as_slice(), &device).reshape([100, 1]);

    (features, y)
}

pub fn log_reg_test() {
    let (features, y) = iris_setosa_versicolor();
    let w = binary_logistic_regression(y.clone(), features.clone());
    let p = predict(features, w);
    println!("{:?}", p.into_data().to_vec::<f32>().unwrap());
}

fn add_bias(features: Tensor<Backend, 2>) -> Tensor<Backend, 2> {
    let f = features.dims()[0];
    let ones = Tensor::<Backend, 2>::ones([f, 1], &Default::default());
    Tensor::<Backend, 2>::cat(vec![features, ones], 1)
}

pub fn predict(features: Tensor<Backend, 2>, weights: Tensor<Backend, 2>) -> Tensor<Backend, 2> {
    sig(add_bias(features), weights)
}

pub fn binary_logistic_regression(
    y: Tensor<Backend, 2>,
    features: Tensor<Backend, 2>,
) -> Tensor<Backend, 2> {
    let features = add_bias(features);
    let f = features.dims()[1];

    let mut weights = Tensor::<Backend, 2>::zeros([f, 1], &Default::default());

    for _ in 0..1000 {
        weights = grad(y.clone(), features.clone(), weights, 0.1);
    }
    weights
}

fn sig(features: Tensor<Backend, 2>, weights: Tensor<Backend, 2>) -> Tensor<Backend, 2> {
    sigmoid(features.matmul(weights))
}

fn grad(
    y: Tensor<Backend, 2>,
    features: Tensor<Backend, 2>,
    weights: Tensor<Backend, 2>,
    learning_rate: f64,
) -> Tensor<Backend, 2> {
    let mut s = sig(features.clone(), weights.clone()).sub(y);
    s = s.mul_scalar(1. / features.dims()[0] as f64);
    s = features.transpose().matmul(s);

    weights.sub(s.mul_scalar(learning_rate))
}
