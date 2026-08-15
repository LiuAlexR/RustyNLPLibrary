use std::{collections::HashMap, hash::Hash};

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
pub fn unigram_creation(vocab_size: usize, corpus: &Vec<usize>) -> Vec<usize> {
    let mut counts: Vec<usize> = vec![0; vocab_size];
    for i in corpus {
        counts[*i] += 1;
    }
    counts
}
pub fn bigram_creation(vocab_size: usize, corpus: &Vec<usize>) -> Vec<HashMap<usize, usize>> {
    // Array that counts the words that appear after the word in the array
    let mut vocab_arr: Vec<HashMap<usize, usize>> = vec![HashMap::new(); vocab_size];
    for i in 0..(corpus.len() - 1) {
        vocab_arr[i].entry(corpus[i + 1]).and_modify(|counter| *counter += 1).or_insert(1);
    }
    vocab_arr
}
pub fn bigram_test(counts: &Vec<HashMap<usize, usize>>, unigram: &Vec<usize>) -> Vec<usize> {
    let mut starting: usize = 376; // I
    let mut res: Vec<usize> = Vec::new();
    for _ in 0..5 {
        let cur_token_bigram = &counts[starting];
        let mut cur_token_prob = unigram[starting] + counts.len();
        let rand_num = rand::random_range(1..(cur_token_prob + 1));
        for i in 0..counts.len() {
            let current_val = match cur_token_bigram.get(&i) {
                Some(x) => x + 1,
                None => 1,
            };
            if cur_token_prob <= current_val {
                res.push(i);
                starting = i;
                break;
            } else {
                cur_token_prob -= current_val;
            }
        }
    }
    res
}