use burn::backend::{Autodiff, Wgpu};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{activation::relu, Distribution, Tensor};

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

pub fn create_random_tensor() -> Tensor<Backend, 2> {
    let device = Default::default();
    let dis = Distribution::Uniform(0., 1.);
    let shape = [VOCAB, DIMENSIONS];

    Tensor::<Backend, 2>::random(shape, dis, &device)
}

pub fn find_derivative(
    target: Tensor<Backend, 2>,
    context: Tensor<Backend, 2>,
    negatives: Option<Vec<Tensor<Backend, 2>>>,
    w: Word,
) -> Tensor<Backend, 2> {
    match w {
        Word::Context => (relu(target.clone() * context) - 1) * target,
        Word::Target => (relu(target.clone() * context) - 1) * target, //TODO(TheSilentIce) impl ∂L/∂w
        Word::Negative => (relu(target.clone() * context) - 1) * target, //TODO(TheSilentIce) impl ∂L/∂negative
    }
}

pub fn start() {
    let a = create_random_tensor();
    let b = create_random_tensor();

    let c = find_derivative(a, b, None, Word::Context);

    println!("dL/dCpos = {}", c.to_data());
}
