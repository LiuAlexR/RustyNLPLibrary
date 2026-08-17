use burn::backend::{Autodiff, Wgpu};
use burn::tensor::activation::sigmoid;
use burn::tensor::{Distribution, Tensor};

pub type Backend = Autodiff<Wgpu>;

pub enum Word {
    Target,
    Context,
    Negative,
}

// According to wikipedia, dimensionality is between 100-1000
// Stack Overflow says 100-300
pub const DIMENSIONS: usize = 3;
pub const VOCAB: usize = 10;

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

pub fn create_random_matrix() -> Tensor<Backend, 2> {
    let device = Default::default();
    let dis = Distribution::Uniform(0., 1.);
    let shape = [VOCAB, DIMENSIONS];
    Tensor::<Backend, 2>::random(shape, dis, &device)
}

pub fn create_random_matrix_custom_dimensions(vocab_size: usize, dimensions: usize) -> Tensor<Backend, 2> {
    let device = Default::default();
    let dis = Distribution::Uniform(0., 1.);
    let shape = [vocab_size, dimensions];
    Tensor::<Backend, 2>::random(shape, dis, &device)
}
/// Calculates loss of target, context word, and negatives
///
/// Page 109 of the book, eq 5.21 is what is implemented
///
/// # Arguments
///
/// * `target` - Target word's tensor
/// * `context` - Context word of target word tensor
/// * `negatives` - Vec of negatively sampled words' tensors
///
/// # Returns
///
/// A `Tensor<Backend,1>` representing the loss
pub fn loss_func(
    target: Tensor<Backend, 1>,
    context: Tensor<Backend, 1>,
    negatives: Vec<Tensor<Backend, 1>>,
) -> Tensor<Backend, 1> {
    let first = sigmoid(target.clone().dot(context)).log();
    let mut second: Tensor<Backend, 1> = Tensor::<Backend, 1>::zeros_like(&target);

    for t in negatives {
        second = second.add(sigmoid(target.clone().dot(t.neg())).log());
    }

    second.add(first).neg()
}

/// Calculates partial derivatives of Loss function
///
/// Partial derivatives of Loss function with respect to
/// target, context word, and each negative word.
/// The partial derviative is picked via the Word enum.
///
/// # Arguments
/// * `target` - Target word's tensor
/// * `context` - Context word of target word tensor
/// * `negatives` - Vec of negatively sampled words' tensors
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

/// Replaces matrix row vec with updated values
///
/// # Arguments
/// * `matrix` - Matrix to update
/// * `new_row` - row with new values
/// * `idx` - index of row vector to update
///
/// # Returns
///
/// A `Tensor<Backend,2>` with the specified index row vector
/// updated
pub fn update_matrix(
    matrix: Tensor<Backend, 2>,
    new_row: Tensor<Backend, 2>,
    idx: usize,
) -> Tensor<Backend, 2> {
    assert!(idx >= 1, "Index must 1-indexed and >= 1");
    assert!(idx <= VOCAB, "Index must be <= VOCAB");
    matrix.slice_assign([idx - 1..idx, 0..DIMENSIONS], new_row)
}

/// Increments value at idx of row vec by 1
///
/// # Arguments
///
/// * `row` - row to be updated
/// * `idx` - index of row to upate
///
/// # Returns
///
/// A `Tensor<Backend,1>` representing row vec with incremented value
pub fn increment_row(row: Tensor<Backend, 1>, idx: usize) -> Tensor<Backend, 1> {
    assert!(idx >= 1, "Index must be 1-indexed and >= 1");
    assert!(idx <= VOCAB, "Index must be <= VOCAB");

    let zero_idx = idx - 1;
    let current = row.clone().slice([zero_idx..zero_idx + 1]);
    let bumped = current + 1.0;
    row.slice_assign([zero_idx..zero_idx + 1], bumped)
}

/// Increments Wij in matrix, where W is word, i is row vec and j is column
///
/// # Arguments
/// * 'matrix' - matrix to update
/// * 'vocab_idx' - index of row vec(Word)
/// * 'co_word_idx' - index of co_word to increment
///
/// # Returns
/// A `Tensor<Backend,2>` represnting 2D matrix
pub fn increment_row_in_matrix(
    matrix: Tensor<Backend, 2>,
    vocab_idx: usize,
    co_word_idx: usize,
) -> Tensor<Backend, 2> {
    assert!(vocab_idx >= 1, "Index must be 1-indexed and >= 1");
    assert!(vocab_idx <= VOCAB, "Index must be <= VOCAB");
    assert!(co_word_idx >= 1, "Index must be 1-indexed and >= 1");
    assert!(co_word_idx <= VOCAB, "Index must be <= VOCAB");

    let row: Tensor<Backend, 1> = matrix.clone().slice([vocab_idx - 1..vocab_idx]).squeeze();
    let bumped_row = increment_row(row, co_word_idx);
    let bumped_2d: Tensor<Backend, 2> = bumped_row.unsqueeze();
    matrix.slice_assign([vocab_idx - 1..vocab_idx], bumped_2d)
}

// test func, to be removed
pub fn start() {
    let mut mat = Tensor::<Backend, 2>::from([[1, 2, 3], [2, 3, 4]]);

    println!("mat = {}", mat);
    mat = increment_row_in_matrix(mat, 1, 1);

    println!("mat = {}", mat);
}
