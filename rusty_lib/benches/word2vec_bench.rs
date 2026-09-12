use criterion::{criterion_group, criterion_main, Criterion};
use rusty_lib::components::tokenizer::{bpe_encoder, bpe_tokenize, text_to_indices};
use rusty_lib::components::w2v::build_model;
use rusty_lib::util;

fn build_bpe_vocab_and_corpus(text: &str, num_tokens: u64) -> (Vec<String>, Vec<usize>) {
    let vocabulary: Vec<String> = bpe_tokenize(text, num_tokens, false);
    let text_owned = text.to_string();
    let encoded: Vec<String> = bpe_encoder(&vocabulary, &text_owned);
    let corpus: Vec<usize> = text_to_indices(&vocabulary, &encoded);
    (vocabulary, corpus)
}

fn analogy_accuracy(
    model: &rusty_lib::components::w2v::Word2Vec,
    analogies: &[(usize, usize, usize, usize)],
) -> f64 {
    let mut correct = 0;
    for &(a, b, cc, expected) in analogies {
        let predicted = model.nearest_to_analogy(a, b, cc);
        if predicted == expected {
            correct += 1;
        }
    }
    correct as f64 / analogies.len() as f64
}

fn run_quality_eval(vocab: &[String], corpus: &[usize]) {
    println!("Real BPE vocab size: {}", vocab.len());
    println!(
        "Real BPE corpus size (total token occurrences): {}",
        corpus.len()
    );

    let (model, _map) = build_model(vocab, 128, 2, 5, 0.025);

    let analogies: Vec<(usize, usize, usize, usize)> = vec![];
    if !analogies.is_empty() {
        let acc = analogy_accuracy(&model, &analogies);
        println!("Analogy accuracy: {:.2}%", acc * 100.0);
    }
}
fn count_pairs(corpus_len: usize, window_size: usize) -> usize {
    if corpus_len <= 2 * window_size {
        return 0;
    }
    let valid_positions = corpus_len - 2 * window_size;
    valid_positions * 2 * window_size
}

fn bench_train_batched(c: &mut Criterion) {
    let full_text = util::retrieve_source("tinystories_sample.txt");
    // small slice — train_naive runs inside criterion's repeated-iteration loop,
    // so this needs to stay fast per iteration, not reflect the full training corpus
    let text: String = full_text.chars().take(200_000).collect();

    let num_tokens = 2000; // match your real num_merges
    let (vocab, corpus) = build_bpe_vocab_and_corpus(&text, num_tokens);
    run_quality_eval(&vocab, &corpus);

    let window_size = 2;
    let k = 5;
    let num_pairs = count_pairs(corpus.len(), window_size);
    let embeddings_per_pair = 2 + k;
    let total_embedding_updates = num_pairs * embeddings_per_pair;

    println!("Corpus length: {}", corpus.len());
    println!("Training pairs generated: {}", num_pairs);
    println!("Embedding updates per pair: {}", embeddings_per_pair);
    println!("Total embedding updates: {}", total_embedding_updates);

    let unigram = vec![1usize; vocab.len()];
    let batch_size = 128usize;

    c.bench_function("word2vec_train_batch128", |b| {
        b.iter(|| {
            let (mut model, _map) = build_model(&vocab, 128, 2, 5, 0.025);
            model.train_naive(&corpus, &unigram, batch_size);
        });
    });
}

criterion_group!(benches, bench_train_batched);
criterion_main!(benches);
