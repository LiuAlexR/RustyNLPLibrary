use std::time::Instant;

// use rusty_lib::components::*;
use rusty_lib::{components::tokenizer, util};
fn main() {
    // let x = util::retrieve_source("liu_hello_world.txt");
    let x = util::retrieve_source("orwell_1984.txt");
    let tokens = 5000;
    let start = Instant::now();
    let y = tokenizer::bpe_tokenize(&x, tokens, true);
    println!("{:?}", y);
    println!(
        "It took {}ms for {} tokens",
        start.elapsed().as_millis(),
        tokens
    );

    // println!("{x}");
    // let _ = util::write_to_storage("test.txt", &"boo\nbooo".to_string());
}
