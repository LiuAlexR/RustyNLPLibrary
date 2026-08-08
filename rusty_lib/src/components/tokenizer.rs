use std::{collections::HashMap, vec::Vec};

/// Byte-Pair Encoding scheme
/// @param corpus - text to tokenize
/// @param num_tokens - maximum number of tokens in vocabulary plus default ASCII
pub fn bpe_tokenize(corpus: &str, num_tokens: u64) -> Vec<String> {
    let mut vocabulary: Vec<String> = (0..128).map(|b: u8| (b as char).to_string()).collect();

    let arr: Vec<char> = corpus.chars().collect();
    for _ in 0..num_tokens {
        let token = combine(&arr, &vocabulary);

        if let Some(token) = token {
            vocabulary.push(token)
        }
    }

    vocabulary
}

fn combine(arr: &[char], vocabulary: &[String]) -> Option<String> {
    let mut map: HashMap<String, u64> = Default::default();

    let len: u64 = arr.len() as u64;
    let mut i = 0;

    let mut token: String = String::default();

    while i < len {
        if arr[i as usize] == ' ' {
            i += 1;
            token.clear();
            continue;
        }

        token.push(arr[i as usize]);

        if has_token(&token, vocabulary) {
            i += 1;
            continue;
        }

        *map.entry(token.clone()).or_insert(0) += 1;
        token.clear();
    }

    if map.is_empty() {
        return None;
    }

    // for ties, returns the last one
    // not deterministic, as pointers are stored anywhere
    let (x, _) = match map.iter().max_by_key(|&(_, y)| y) {
        None => panic!("Error finding max of token map"),
        Some((x, y)) => (x, y),
    };

    Some(x.clone())
}

// may want to use a better algorithm
fn has_token(token: &str, vocabulary: &[String]) -> bool {
    for n in vocabulary {
        if n == token {
            return true;
        }
    }

    false
}

// TODO(LiuAlexR) - implement function to tokenize future inputs
