use criterion::{criterion_group, criterion_main, Criterion};
use rusty_lib::components::tokenizer::bpe_tokenize;
use rusty_lib::*;
use std::hint::black_box;

// to add future benchmarks, follow this format
// add the new func, to criterion_group!
// black-box prevents compiler optimizations

fn bench_bpe_tokenize(c: &mut Criterion) {
    let x = util::retrieve_source("orwell_1984.txt");
    let tokens = 1000;

    c.bench_function("bpe_tokenize orwell 5000", |b| {
        b.iter(|| black_box(bpe_tokenize(&x, tokens, true)))
    });
}

criterion_group!(benches, bench_bpe_tokenize);
criterion_main!(benches);
