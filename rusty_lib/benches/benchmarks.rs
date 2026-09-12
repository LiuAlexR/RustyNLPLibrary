use criterion::{criterion_group, criterion_main, Criterion};
use rusty_lib::components::tokenizer::{bpe_encoder, bpe_tokenize};
use rusty_lib::math::{create_random_vector, find_derivative};
use rusty_lib::*;
use std::hint::black_box;

// to add future benchmarks, follow this format
// add the new func, to criterion_group!
// black-box prevents compiler optimizations

fn bench_bpe_tokenize(c: &mut Criterion) {
    let full_text = util::retrieve_source("tinystories_sample.txt");
    let x: String = full_text.chars().take(200_000).collect(); // ~200KB slice, not the full 20MB
    let tokens = 2000; // match whatever num_merges you actually train with

    c.bench_function("bpe_tokenize tinystories 200k", |b| {
        b.iter(|| black_box(bpe_tokenize(&x, tokens, true)))
    });
}

fn bench_bpe_encoder(c: &mut Criterion) {
    let full_text = util::retrieve_source("tinystories_sample.txt");
    let text: String = full_text.chars().take(200_000).collect();
    let vocab = bpe_tokenize(&text, 2000, false);

    c.bench_function("bpe_encoder 200k", |b| {
        b.iter(|| bpe_encoder(&vocab, &text))
    });
}

fn bench_find_derivative(c: &mut Criterion) {
    let N = 400;
    let mut v = vec![];
    for _ in 0..N {
        v.push(create_random_vector(100));
    }

    c.bench_function("bench: find_derivative", |b| {
        b.iter(|| {
            black_box(find_derivative(
                create_random_vector(100),
                create_random_vector(100),
                v.clone(),
                math::Word::Negative,
            ))
        })
    });
}

criterion_group!(token_benches, bench_bpe_tokenize, bench_bpe_encoder);
criterion_group!(math_benches, bench_find_derivative);
criterion_main!(token_benches, math_benches);
