use criterion::{criterion_group, criterion_main, Criterion};
use rusty_lib::components::tokenizer::bpe_tokenize;
use rusty_lib::math::{create_random_tensor, find_derivative};
use rusty_lib::*;
use std::hint::black_box;

// to add future benchmarks, follow this format
// add the new func, to criterion_group!
// black-box prevents compiler optimizations

fn bench_bpe_tokenize(c: &mut Criterion) {
    let x = util::retrieve_source("orwell_1984.txt");
    const TOKENS: u64 = 5000;

    c.bench_function("bpe_tokenize orwell 5000", |b| {
        b.iter(|| black_box(bpe_tokenize(&x, TOKENS, true)))
    });
}

fn bench_find_derivative(c: &mut Criterion) {
    const N: u64 = 400;
    let mut v = vec![];
    for _ in 0..N {
        v.push(create_random_tensor());
    }

    c.bench_function("bench: find_derivative", |b| {
        b.iter(|| {
            black_box(find_derivative(
                create_random_tensor(),
                create_random_tensor(),
                v.clone(),
                math::Word::Negative,
            ))
        })
    });
}

criterion_group!(token_benches, bench_bpe_tokenize);
criterion_group!(math_benches, bench_find_derivative);
criterion_main!(token_benches, math_benches);
