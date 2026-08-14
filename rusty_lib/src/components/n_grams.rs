use std::collections::HashMap;

use crate::components::tokenizer::bpe_tokenize;

pub fn print_test() -> () {
    let x = "Kareem Abdul-Jabbar was the Finals MVP of the 1971 NBA FINALS, winning against the Los Angeles Lakers.";
    let vocab: Vec<String> = Vec::new();
    let mut unigram: HashMap<String, usize> = HashMap::new();
    for i in &vocab {
        unigram.insert(i.to_string(), x.matches(i).count());
    }
    //let v = bpe_tokenize(x, 200);
    //println!("{:?}", v);
}
pub fn bigram_creation(corpus: &str, vocab: &Vec<String>) -> () {
    
}
