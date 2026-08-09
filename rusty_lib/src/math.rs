// create_matrix from vector/array
// gradient descent
//

use burn::backend::{Autodiff, Wgpu};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Distribution, Numeric, Shape, Tensor};

type Backend = Autodiff<Wgpu>;
// According to wikipedia, dimensionality is between 100-1000
// Stack Overflow says 100-300
const DIMENSIONS: usize = 2;

// takes ownership
pub fn take_partial<D: AutodiffBackend>(a: Tensor<D, DIMENSIONS>, b: Tensor<D, DIMENSIONS>) {
    let a = a.require_grad();
    let b = b.require_grad();
    let y = ((a.clone() * b.clone()) + a.clone()).backward();

    let grad_a = a.grad(&y).unwrap();
    let grad_b = b.grad(&y).unwrap();

    println!("dy/da = {}", grad_a.to_data());
    println!("dy/db = {}", grad_b.to_data());
}

pub fn start() {
    // let device = Backend::default();
    let device = Default::default();
    let dis = Distribution::Uniform(0., 1.);

    let shape = [50_000, 300];

    let tensor = Tensor::<Backend, DIMENSIONS>::random(shape, dis, &device);
    println!("{tensor}");
}
