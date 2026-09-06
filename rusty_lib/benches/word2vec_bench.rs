use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rusty_lib::components::w2v::build_model;

fn make_vocab(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("w{i}")).collect()
}

fn make_corpus(len: usize, vocab_size: usize) -> Vec<usize> {
    (0..len).map(|i| i % vocab_size).collect()
}

fn bench_train_batched(c: &mut Criterion) {
    let mut group = c.benchmark_group("word2vec_train_batched");
    group.sample_size(10);
    let vocab_size = 10_000;
    let vocab = make_vocab(vocab_size);
    let corpus = make_corpus(2_000, vocab_size);
    let unigram = vec![1usize; vocab_size];
    for &batch_size in &[1usize, 8, 32, 128] {
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &batch_size,
            |b, &bs| {
                b.iter(|| {
                    let mut model = build_model(&vocab, 128, 2, 5, 0.025);
                    model.train_naive(&corpus, &unigram, bs);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_train_batched);
criterion_main!(benches);
