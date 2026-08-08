use std::{collections::HashMap, vec::Vec};

// corpus is going to be just the direct txt file into a string
// num tokens - number of tokens we want to add atop the default ascii chars
//
// split corpus into chars
//
// a s d f => as d f => asd f
pub fn tokenize(corpus: &str, num_tokens: u64) -> Vec<&str> {
    let mut vocabulary: Vec<&str> = vec![
        " ", "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1",
        "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@", "A", "B", "C",
        "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
        "V", "W", "X", "Y", "Z", "[", "\\", "]", "^", "_", "`", "a", "b", "c", "d", "e", "f", "g",
        "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y",
        "z", "{", "|", "}", "~",
    ];

    let arr: Vec<&str> = corpus.split("").collect();

    for i in 0..num_tokens {
        // iterate through arr
        // for combining tokens, use a while loop alongside find_token method to see if token exists
        // then, once you have an existing token, add the next char to it, then update map
        //
        // once you do that for arr, return most frequent token, on ties, we'll just add all of them
        // let max = m.iter().max();
        // make sure that you do not combine words with a space between them
        //
        // add token to vocab and clear the map
        // m.clear()
    }

    vocabulary
}

fn combine<'a>(arr: &'a [&str]) -> &'a str {
    let mut m: HashMap<&str, u64> = Default::default();

    let len: u64 = arr.len() as u64;
    let mut i = 0;

    while i < len {
        i += 1
    }

    let (x, _) = match m.iter().max_by_key(|&(_, y)| y) {
        None => panic!("Error finding max of token map"),
        Some((x, y)) => (x, y),
    };

    x
}

// may want to use a better algorithm
fn find_token(token: &str, vocabulary: &[&str]) -> bool {
    for n in vocabulary {
        if *n == token {
            return true;
        }
    }

    false
}
