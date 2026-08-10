use burn::backend::{Autodiff, Wgpu};
use burn::tensor::activation::sigmoid;
use burn::tensor::{Distribution, Tensor};

type Backend = Autodiff<Wgpu>;

enum Word {
    Target,
    Context,
    Negative,
}

// According to wikipedia, dimensionality is between 100-1000
// Stack Overflow says 100-300
const DIMENSIONS: usize = 3;
const VOCAB: usize = 10;

/// Creates random tensor
///
/// Generates a 1D tensor with random initialized values
///
/// # Returns
///
/// a `Tensor<Backend,1>` with random initialized values
pub fn create_random_tensor() -> Tensor<Backend, 1> {
    let device = Default::default();
    let dis = Distribution::Uniform(0., 1.);
    let shape = [DIMENSIONS];

    Tensor::<Backend, 1>::random(shape, dis, &device)
}

/// Calculates partial derivatives of Loss function
///
/// Partial derivatives of Loss function with respect to
/// target, context word, and each negative word.
/// The partial derviative is picked via the Word enum.
///
/// # Returns
///
/// A `Vec<Tensor<Backend,1>>` that has all partial derivative values
/// For target and context, it will still return a `Vec<Tensor<Backend,1>>`
/// with one element
pub fn find_derivative(
    target: Tensor<Backend, 1>,
    context: Tensor<Backend, 1>,
    negatives: Vec<Tensor<Backend, 1>>,
    w: Word,
) -> Vec<Tensor<Backend, 1>> {
    match w {
        Word::Context => vec![((sigmoid(target.clone().dot(context)) - 1) * target)],
        Word::Target => get_negatives(target.clone(), negatives.clone(), true),
        Word::Negative => {
            let x = get_negatives(target.clone(), negatives.clone(), false);
            let mut a: Tensor<Backend, 1> = Tensor::zeros_like(&target);

            for t in x {
                a = a.add(t);
            }
            vec![a]
        }
    }
}

// calculates partial derivative of negatives and multiplies
// with either target or the negative vector, dependent on with_target
fn get_negatives(
    target: Tensor<Backend, 1>,
    negatives: Vec<Tensor<Backend, 1>>,
    with_target: bool,
) -> Vec<Tensor<Backend, 1>> {
    let mut v = vec![];

    for t in negatives {
        if with_target {
            let result = target.clone().dot(t.clone()).mul(target.clone());
            v.push(result);
        } else {
            let result = target.clone().dot(t.clone()).mul(t);
            v.push(result);
        }
    }

    v
}

// test func, to be removed
pub fn start() {
    let a = create_random_tensor();
    let b = create_random_tensor();
    let x = vec![a.clone(), b.clone()];

    let c = find_derivative(a, b, x, Word::Negative).pop().unwrap();

    println!("dL/dCpos ={:?}", c.to_data());
}
