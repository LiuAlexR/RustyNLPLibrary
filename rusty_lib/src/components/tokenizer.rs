use std::{
    collections::{HashMap, VecDeque},
    vec::Vec,
};
use crate::math::{increment, Backend, VOCAB};
use burn::Tensor;

/// Byte-Pair Encoding scheme
///
/// Implements Byte-Pair Encoding for tokenization on a given corpus.
///
/// # Arguments
///
/// * `corpus` - Text to tokenize
/// * `num_tokens` - maximum vocabulary size alongside ASCII chars
/// * `only_new` - whether to return the full vocabulary or just the new tokens
///
/// # Returns
///
/// A `Vec<String>` containing learned tokens(by default, includes ASCII chars)
///
/// # Examples
///
/// `let vocab = bpe_tokenize(corpus, 5000, true);`
pub fn bpe_tokenize(corpus: &str, num_tokens: u64, only_new: bool) -> Vec<String> {
    let mut vocabulary: Vec<String> = (0..128).map(|b: u8| (b as char).to_string()).collect();

    let mut new_vocab: Vec<String> = Vec::default();
    let arr: Vec<char> = corpus.chars().collect();
    let (token, mut map) = combine(&arr);

    if only_new {
        new_vocab.push(token.clone());
    }
    vocabulary.push(token);

    for _ in 1..num_tokens {
        let token = combine_with_index(&arr, &vocabulary, &mut map);

        if let Some(token) = token {
            if only_new {
                new_vocab.push(token.clone());
            }
            // println!("{}", token);
            vocabulary.push(token);
        }
    }

    if only_new {
        new_vocab
    } else {
        vocabulary
    }
}

// takes last token added and goes through its indices vector to create new tokens
fn combine_with_index(
    arr: &[char],
    vocabulary: &[String],
    map: &mut HashMap<String, (u64, Vec<u64>)>,
) -> Option<String> {
    if map.is_empty() {
        return None;
    }

    let len = arr.len() as u64;

    let token = vocabulary.last().unwrap();

    let (_, indices) = map.get(token).unwrap();
    let indices = indices.clone();

    for i in indices {
        if i + 1 == len || arr[(i + 1) as usize] == '\n' || arr[(i + 1) as usize] == ' ' {
            continue;
        }

        let largest_token = find_largest_token(arr, i + 1, vocabulary);

        let x = token.to_owned() + &largest_token;
        let index = i + largest_token.len() as u64;

        map.entry(x.clone())
            .and_modify(|(count, indices)| {
                *count += 1;
                indices.push(index);
            })
            .or_insert((1, vec![index]));
    }

    map.remove(token);
    let x = map
        .iter()
        .max_by_key(|(_, (count, _))| count)
        .map(|(x, _)| x.clone())
        .expect("Error finding max");

    Some(x)
}

// Generate all possible two letter tokens
// TODO spaces can be the beginning of a token but not end
fn combine(arr: &[char]) -> (String, HashMap<String, (u64, Vec<u64>)>) {
    let mut map: HashMap<String, (u64, Vec<u64>)> = Default::default();
    let len = arr.len();

    for i in 0..len.saturating_sub(1) {
        let a = arr[i];
        let b = arr[i + 1];

        if a == '\n' || b == ' ' || b == '\n' {
            continue;
        }

        let mut token = String::with_capacity(2);
        token.push(a);
        token.push(b);

        let index = (i + 1) as u64;

        map.entry(token)
            .and_modify(|(count, indices)| {
                *count += 1;
                indices.push(index);
            })
            .or_insert((1, vec![index]));
    }
    let (x, _) = map
        .iter()
        .max_by_key(|(_, (count, _))| count)
        .expect("Error finding max");

    (x.clone(), map)
}

// |--TARGET--|
// window - defined as number of words before and after
// so if window = 2, first 2 words before and after target
//
// 5 == target 3,4 6,7
fn count_word<'a>(
    target_idx: u64,
    map: &mut HashMap<&'a str, (u64, Tensor<Backend, 1>)>,
    input: &'a [String],
    context_window: u64,
) {
    let s: &str = &input[target_idx as usize];
    let (idx, mut t) = map.get(s).unwrap().clone();
    for i in (target_idx - context_window)..(target_idx - 1) {
        let (context_idx, _) = map.get(&input[i as usize] as &str).unwrap();
        t = increment(t.clone(), *context_idx as usize);
    }

    for i in (target_idx + 1)..(target_idx + context_window) {
        let (context_idx, _) = map.get(&input[i as usize] as &str).unwrap();
        t = increment(t.clone(), *context_idx as usize);
    }

    map.insert(s, (idx, t));
}

// create a hashmap H<&String, (u64, Tensor::<Backend,2>> where tuple is (idx in vocab, tensor
// representing all words)
// for each word, update H
// then at end, create tensor and update matrix
pub fn co_occurence(input: &[String], vocab: &[String], context_window: u64) -> Tensor<Backend, 2> {
    assert!(context_window >= 1, "Context window must be at least 1");

    let device = Default::default();
    let ten = Tensor::<Backend, 2>::zeros([VOCAB, VOCAB], &device);

    let mut map: HashMap<&str, (u64, Tensor<Backend, 1>)> = vocab
        .iter()
        .enumerate()
        .map(|(idx, s)| {
            (
                s.as_str(),
                (idx as u64, Tensor::<Backend, 1>::zeros([VOCAB], &device)),
            )
        })
        .collect();

    ten
}

fn find_largest_token(arr: &[char], mut idx: u64, vocabulary: &[String]) -> String {
    let mut token = String::default();
    let len = arr.len() as u64;

    token.push(arr[idx as usize]);

    loop {
        let x = vocabulary.iter().find(|&y| *y == token);
        if x.is_none() {
            break;
        }
        idx += 1;

        if idx == len || arr[idx as usize] == '\n' || arr[idx as usize] == ' ' {
            return token;
        }

        token.push(arr[idx as usize]);
    }
    token.pop();
    token
}
pub fn bpe_encoder(vocabulary: &Vec<String>, text: &String) -> Vec<String> {
    // Pairs each string with corresponding idx in vocab
    // Earlier strings in vocab occur more than later ones,
    // thus natural priority arises
    let rank: HashMap<&str, usize> = vocabulary
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

    let words: Vec<&str> = text.split_whitespace().collect();
    let mut result = Vec::new();

    for w in words {
        let mut tokens: Vec<String> = std::iter::once(" ".to_string())
            .chain(w.chars().map(String::from))
            .collect();

        loop {
            // find the single best (lowest-rank) merge available in this word
            let mut best: Option<(usize, usize)> = None; // (rank, position)
            for k in 0..tokens.len().saturating_sub(1) {
                let mut pair = tokens[k].clone();
                pair.push_str(&tokens[k + 1]);

                // If pair exists in rank map and if the priority is higher
                if let Some(&r) = rank.get(pair.as_str())
                    && best.is_none_or(|(br, _)| r < br) 
                {
                        best = Some((r, k));
                }
            }
            match best {
                Some((_, k)) => {
                    let merged = tokens[k].clone() + &tokens[k + 1];
                    tokens[k] = merged;
                    tokens.remove(k + 1);
                }
                None => break,
            }
        }
        result.extend(tokens);
    }
    result
}

pub fn bpe_encoder_a(vocabulary: &Vec<String>, text: &String) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut broken_words: Vec<Vec<String>> = Vec::new();
    for i in &words {
        let mut chars: Vec<String> = (*i).chars().map(String::from).collect();
        chars.insert(0, " ".to_string());
        broken_words.push(chars);
    }
    for i in vocabulary {
        for j in 0..broken_words.len() {
            for k in 0..(broken_words[j].len()-1) {
                if k >= broken_words[j].len()-1 {
                    break;
                }
                let mut temp = broken_words[j][k].clone();
                temp.push_str(&broken_words[j][k + 1].clone());
                if temp == *i {
                    broken_words[j].remove(k);
                    broken_words[j].remove(k);
                    broken_words[j].insert(k, temp);
                }
                
            }
        }
    }
    let mut res: Vec<String> = Vec::new();
    for i in broken_words {
        for j in i {
            res.push(j);
        }
    }
    res
}
/// Turns the vector of String tokens into a vector of integer tokens, by the token's index
pub fn text_to_indices(vocabulary: &Vec<String>, encoded: &Vec<String>) -> Vec<usize> {
    let rank: HashMap<&str, usize> = vocabulary
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let mut new_tokens: Vec<usize> = Vec::new();
    for i in encoded {
        new_tokens.push(rank[i.as_str()]);
    }
    new_tokens
}
pub fn indices_to_text(vocabulary: &Vec<String>, encoded: &Vec<usize>) -> Vec<String> {
    
    let mut new_tokens: Vec<String> = Vec::new();
    for i in encoded {
        new_tokens.push(vocabulary[*i].clone());
    }
    new_tokens
}
pub fn usize_to_token(vocabulary: &Vec<String>, num: usize) -> &str {
    &vocabulary[num]
}