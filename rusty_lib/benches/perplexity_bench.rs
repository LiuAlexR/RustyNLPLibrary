use criterion::{criterion_group, criterion_main, Criterion};
use rusty_lib::components::attention::{evaluate_perplexity, init_transformer, train, Transformer};
use rusty_lib::components::tokenizer::{bpe_encoder, bpe_tokenize};
use rusty_lib::components::w2v::build_vocab_map;
use rusty_lib::math::create_random_matrix;
use rusty_lib::util;

fn setup_trained_model() -> (
    Transformer,
    Vec<String>,
    std::collections::HashMap<String, i64>,
) {
    let full_text = util::retrieve_source("tinystories_sample.txt");
    let text: String = full_text.chars().take(500_000).collect(); // small slice, kept fast

    let split = (text.len() as f64 * 0.9) as usize;
    let train_text = &text[..split];
    let val_text = &text[split..];

    let vocab = bpe_tokenize(&text, 1000, false);
    let map = build_vocab_map(&vocab);
    let train_tokens = bpe_encoder(&vocab, &train_text.to_string());
    let val_tokens = bpe_encoder(&vocab, &val_text.to_string());

    let d = 128;
    let e = create_random_matrix(vocab.len(), d).require_grad();
    let mut model = init_transformer(
        2,
        4,
        d,
        512,
        e,
        "<PADD>".to_string(),
        map["<PADD>"] as usize,
    );

    for _ in 0..3 {
        train(&train_tokens, &mut model, 32, 16, &map, 1e-4);
    }

    (model, val_tokens, map)
}

fn bench_perplexity_eval(c: &mut Criterion) {
    let (model, val_tokens, map) = setup_trained_model();

    // one-time quality metric — not what's being timed
    let ppl = evaluate_perplexity(&val_tokens, &model, 32, 16, &map);
    println!("Validation perplexity: {ppl:.2}");

    // what's actually worth benchmarking: evaluation throughput
    c.bench_function("evaluate_perplexity", |b| {
        b.iter(|| evaluate_perplexity(&val_tokens, &model, 32, 16, &map));
    });
}

criterion_group!(benches, bench_perplexity_eval);
criterion_main!(benches);
