use crate::math::Backend;
use burn::{
    tensor::activation::{sigmoid, softmax},
    Tensor,
};

type Activator =
    fn(features: Tensor<Backend, 2>, weights: Tensor<Backend, 2>) -> Tensor<Backend, 2>;

// Add virginica (label 2), 50 rows, to make it 3-class
pub fn iris_three_class() -> (Tensor<Backend, 2>, Tensor<Backend, 2>) {
    let device = Default::default();
    let (features2, y2) = iris_setosa_versicolor(); // setosa + versicolor, 100 rows

    let virginica_data: Vec<f32> = vec![
        6.3, 3.3, 6.0, 2.5, 5.8, 2.7, 5.1, 1.9, 7.1, 3.0, 5.9, 2.1, 6.3, 2.9, 5.6, 1.8, 6.5, 3.0,
        5.8, 2.2, 7.6, 3.0, 6.6, 2.1, 4.9, 2.5, 4.5, 1.7, 7.3, 2.9, 6.3, 1.8, 6.7, 2.5, 5.8, 1.8,
        7.2, 3.6, 6.1, 2.5, 6.5, 3.2, 5.1, 2.0, 6.4, 2.7, 5.3, 1.9, 6.8, 3.0, 5.5, 2.1, 5.7, 2.5,
        5.0, 2.0, 5.8, 2.8, 5.1, 2.4, 6.4, 3.2, 5.3, 2.3, 6.5, 3.0, 5.5, 1.8, 7.7, 3.8, 6.7, 2.2,
        7.7, 2.6, 6.9, 2.3, 6.0, 2.2, 5.0, 1.5, 6.9, 3.2, 5.7, 2.3, 5.6, 2.8, 4.9, 2.0, 7.7, 2.8,
        6.7, 2.0, 6.3, 2.7, 4.9, 1.8, 6.7, 3.3, 5.7, 2.1, 7.2, 3.2, 6.0, 1.8, 6.2, 2.8, 4.8, 1.8,
        6.1, 3.0, 4.9, 1.8, 6.4, 2.8, 5.6, 2.1, 7.2, 3.0, 5.8, 1.6, 7.4, 2.8, 6.1, 1.9, 7.9, 3.8,
        6.4, 2.0, 6.4, 2.8, 5.6, 2.2, 6.3, 2.8, 5.1, 1.5, 6.1, 2.6, 5.6, 1.4, 7.7, 3.0, 6.1, 2.3,
        6.3, 3.4, 5.6, 2.4, 6.4, 3.1, 5.5, 1.8, 6.0, 3.0, 4.8, 1.8, 6.9, 3.1, 5.4, 2.1, 6.7, 3.1,
        5.6, 2.4, 6.9, 3.1, 5.1, 2.3, 5.8, 2.7, 5.1, 1.9, 6.8, 3.2, 5.9, 2.3, 6.7, 3.3, 5.7, 2.5,
        6.7, 3.0, 5.2, 2.3, 6.3, 2.5, 5.0, 1.9, 6.5, 3.0, 5.2, 2.0, 6.2, 3.4, 5.4, 2.3, 5.9, 3.0,
        5.1, 1.8,
    ];

    let f3 = Tensor::<Backend, 1>::from_floats(virginica_data.as_slice(), &device).reshape([50, 4]);
    let y3 = Tensor::<Backend, 1>::from_floats(vec![2.0; 50].as_slice(), &device).reshape([50, 1]);

    (
        Tensor::cat(vec![features2, f3], 0),
        Tensor::cat(vec![y2, y3], 0),
    )
}

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

pub fn log_reg_test(act: Activator, num_classes: usize) {
    let (features, y) = iris_three_class();
    let w = logistic_regression(
        y.clone().reshape([features.dims()[0]]).one_hot(num_classes),
        features.clone(),
        act,
        num_classes,
    );
    let p = predict(features, w, act);
    println!("{:?}", p.into_data().to_vec::<f32>().unwrap());
}

fn add_bias(features: Tensor<Backend, 2>) -> Tensor<Backend, 2> {
    let f = features.dims()[0];
    let ones = Tensor::<Backend, 2>::ones([f, 1], &Default::default());
    Tensor::<Backend, 2>::cat(vec![features, ones], 1)
}

pub fn predict(
    features: Tensor<Backend, 2>,
    weights: Tensor<Backend, 2>,
    act: Activator,
) -> Tensor<Backend, 2> {
    act(add_bias(features), weights)
}

pub fn logistic_regression(
    y: Tensor<Backend, 2>,
    features: Tensor<Backend, 2>,
    act: Activator,
    num_classes: usize,
) -> Tensor<Backend, 2> {
    let features = add_bias(features);
    let f = features.dims()[1];

    let mut weights = Tensor::<Backend, 2>::zeros([f, num_classes], &Default::default());

    for _ in 0..1000 {
        weights = grad(y.clone(), features.clone(), weights, 0.1, act);
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
    act: Activator,
) -> Tensor<Backend, 2> {
    let mut s = act(features.clone(), weights.clone()).sub(y);
    s = s.mul_scalar(1. / features.dims()[0] as f64);
    s = features.transpose().matmul(s);

    weights.sub(s.mul_scalar(learning_rate))
}

pub fn sm(features: Tensor<Backend, 2>, weights: Tensor<Backend, 2>) -> Tensor<Backend, 2> {
    softmax(features.matmul(weights), 1)
}
