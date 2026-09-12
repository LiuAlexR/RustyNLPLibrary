use std::time::Duration;

use burn::tensor::{Distribution, Tensor};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use rusty_lib::components::attention::{init_transformer, transformer_forward_pass, Transformer};
use rusty_lib::math::{create_random_matrix, Backend};

fn build_test_model(
    vocab_size: usize,
    d: usize,
    ff: usize,
    num_blocks: usize,
    num_heads: usize,
) -> Transformer {
    let e = create_random_matrix(vocab_size, d).require_grad();

    init_transformer(
        num_blocks,
        num_heads,
        d,
        ff,
        e,
        "<padd>".to_string(),
        vocab_size - 1,
    )
}

fn bench_transformer_forward(c: &mut Criterion) {
    let d = 128;
    let ff = 512;
    let vocab_size = 2000;
    let num_blocks = 2;
    let num_heads = 4;

    let mut group = c.benchmark_group("transformer_forward");

    // Keep the benchmark workload reasonable.
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    for &(batch, len) in &[
        (8usize, 16usize),
        (8usize, 32usize),
        (8usize, 64usize),
        (16usize, 32usize),
    ] {
        let model = build_test_model(vocab_size, d, ff, num_blocks, num_heads);

        let x = Tensor::<Backend, 3>::random(
            [batch, len, d],
            Distribution::default(),
            &Default::default(),
        );

        group.bench_with_input(
            BenchmarkId::new("batch_len", format!("{batch}x{len}")),
            &(batch, len),
            |b, _| {
                b.iter(|| transformer_forward_pass(x.clone(), &model));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_transformer_forward);
criterion_main!(benches);
