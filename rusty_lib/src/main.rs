use std::io::{self, Write};
use std::time::Instant;

use rusty_lib::{
    components::{
        attention::{init_transformer, predict, train, Transformer},
        n_grams::unigram_creation,
        tokenizer::{bpe_encoder, bpe_tokenize, text_to_indices},
        w2v::build_model,
    },
    util::retrieve_source,
};

fn main() {
    let corpus: String = retrieve_source("orwell_1984.txt");

    let num_merges: u64 = 2000;
    let d: usize = 64;
    let ff: usize = 256;
    let num_blocks: usize = 2;
    let num_heads: usize = 4;
    let context_window: usize = 64;
    let batch_size: usize = 64;
    let learning_rate: f64 = 1e-3;
    let num_epochs: usize = 5;

    let w2v_window: usize = 5;
    let w2v_negatives: usize = 5;
    let w2v_lr: f64 = 0.025;
    let w2v_batch_size: usize = 512;

    let start = Instant::now();
    // 1. Build vocab, tokenize
    let vocab: Vec<String> = bpe_tokenize(&corpus, num_merges, false);
    let tokens: Vec<String> = bpe_encoder(&vocab, &corpus);
    let elapsed = start.elapsed();

    println!("Took {:?} s to tokenize", start.elapsed().as_secs());

    // 2. Pretrain embeddings with Word2Vec — this also gives us the vocab map
    let corpus_indices: Vec<usize> = text_to_indices(&vocab, &tokens);
    let unigram: Vec<usize> = unigram_creation(vocab.len(), &corpus_indices);

    let (mut w2v, map) = build_model(&vocab, d, w2v_window, w2v_negatives, w2v_lr);
    w2v.train_naive(&corpus_indices, &unigram, w2v_batch_size);

    let start = Instant::now();
    let e = w2v.embedding_matrix().require_grad();

    println!(
        "Took {:?} s to create embeddings",
        start.elapsed().as_secs()
    );
    let pad_token = "<PADD>".to_string();
    let pad_id = *map.get(&pad_token).expect("pad token missing from vocab") as usize;

    // 3. Model init, seeded with pretrained embeddings
    let mut model: Transformer =
        init_transformer(num_blocks, num_heads, d, ff, e, pad_token, pad_id);

    // 4. Train transformer
    for epoch in 0..num_epochs {
        let start = Instant::now();
        train(
            &tokens,
            &mut model,
            context_window,
            batch_size,
            &map,
            learning_rate,
        );
        println!("epoch {epoch} done");
        println!("Took {:?} s", start.elapsed().as_secs());
    }

    println!("Training complete. Enter a prompt (or 'quit' to exit):");

    // 5. Interactive generation loop
    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() {
            println!("Failed to read input, try again.");
            continue;
        }

        let input = input.trim();
        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            break;
        }
        if input.is_empty() {
            continue;
        }

        let prompt_tokens = bpe_encoder(&vocab, &input.to_string());
        let output = predict(&prompt_tokens, context_window, &map, &model, &vocab, 1);
        println!("{output}");
    }
}
