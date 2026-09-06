use burn::tensor::{Distribution, Tensor};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rusty_lib::components::neural_net::{forward_pass, one_pass, use_relu};
use rusty_lib::math::{create_random_matrix_custom_dimensions, Backend};

const INPUT_DIM: usize = 301; // window*dim + bias — set to your real value
const HIDDEN: usize = 3;

fn bench_forward(c: &mut Criterion) {
    let device = Default::default();
    let mut group = c.benchmark_group("nn_forward");

    for &vocab_size in &[1_000usize, 10_000, 50_000] {
        let x = Tensor::<Backend, 2>::random([1, INPUT_DIM], Distribution::Default, &device);
        let w = create_random_matrix_custom_dimensions(INPUT_DIM, HIDDEN);
        let u = create_random_matrix_custom_dimensions(HIDDEN, vocab_size);

        group.bench_with_input(
            BenchmarkId::new("vocab", vocab_size),
            &vocab_size,
            |b, _| {
                b.iter(|| forward_pass(x.clone(), w.clone(), u.clone(), use_relu));
            },
        );
    }
    group.finish();
}

fn bench_train_step(c: &mut Criterion) {
    let device = Default::default();
    let mut group = c.benchmark_group("nn_train_step");

    for &vocab_size in &[1_000usize, 10_000, 50_000] {
        let x = Tensor::<Backend, 2>::random([1, INPUT_DIM], Distribution::Default, &device);
        let y = Tensor::<Backend, 1>::zeros([vocab_size], &device);
        let w = create_random_matrix_custom_dimensions(INPUT_DIM, HIDDEN).require_grad();
        let u = create_random_matrix_custom_dimensions(HIDDEN, vocab_size).require_grad();

        group.bench_with_input(
            BenchmarkId::new("vocab", vocab_size),
            &vocab_size,
            |b, _| {
                b.iter(|| one_pass(x.clone(), y.clone(), w.clone(), u.clone(), use_relu));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_forward, bench_train_step);
criterion_main!(benches);
