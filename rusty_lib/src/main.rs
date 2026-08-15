#![warn(unused_extern_crates)]
use std::time::Instant;

// use rusty_lib::components::*;
use rusty_lib::{components::tokenizer, components::n_grams, util};
fn main() {
    // let x = util::retrieve_source("liu_hello_world.txt");
    let x = util::retrieve_source("orwell_1984.txt");
    // let tokens = 5000;
    // let y = tokenizer::bpe_tokenize(&x, tokens, false);
    // let _ = util::write_to_storage("orwell_token.txt", &util::vec_to_string("|", &y));

    let vocab = util::read_stored_token("orwell_token.txt");
    let t = Instant::now();
    let tokens = tokenizer::bpe_encoder(&vocab, &x);
    let end = Instant::now() - t;
    println!("Took {} ms!", end.as_millis());
    let index_tokens = tokenizer::text_to_indices(&vocab, &tokens);
    // let test = "I";
    // let tokenized = tokenizer::bpe_encoder(&tokens, &test.to_string());
    // println!("{:?}", tokenized);
    // let tokenizeda = tokenizer::bpe_encoder_a(&tokens, &test.to_string());
    // let int_tokens: Vec<usize> = tokenizer::text_to_indices(&tokens, &tokenizeda);
    let unigram = n_grams::unigram_creation(vocab.len(), &index_tokens);
    let bigram = n_grams::bigram_creation(vocab.len(), &index_tokens);
    let out = n_grams::bigram_test(&bigram, &unigram);
    let out_text: Vec<&str> = out.into_iter().map(|x| tokenizer::usize_to_token(&vocab, x)).collect();
    println!("{:?}", out_text);
}
