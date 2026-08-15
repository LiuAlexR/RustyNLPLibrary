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
    for i in 0..(vocab_arr.len() - 1) {
        vocab_arr[corpus[i]].entry(corpus[i + 1]).and_modify(|counter| *counter += 1).or_insert(1);
    }
    vocab_arr
}
pub fn bigram_test(counts: &Vec<HashMap<usize, usize>>, unigram: &Vec<usize>) -> Vec<usize> {
    let mut starting: usize = 376; // I
    let mut res: Vec<usize> = Vec::new();
    for _ in 0..5 {
        let cur_token_bigram = &counts[starting];
        let mut cur_token_prob: isize = (unigram[starting] + counts.len()) as isize;
        let rand_num: isize = rand::random_range(0..(cur_token_prob as usize + 1)) as isize;
        let mut i: usize = 0;
        while cur_token_prob >= 0 {
            let current_val = match cur_token_bigram.get(&i) {
                Some(x) => x + 1,
                None => 1,
            };
            cur_token_prob -= current_val as isize;
            i = i + 1;
        }
        res.push(i);
    }
    res
}