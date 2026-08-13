use std::{collections::{HashMap, VecDeque}, vec::Vec};

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

    let len: u64 = arr.len() as u64;
    let mut i = 0;

    let mut token: String = String::default();

    while i < len {
        if arr[i as usize] == ' ' || arr[i as usize] == '\n' {
            i += 1;
            token.clear();
            continue;
        }

        token.push(arr[i as usize]);

        if token.len() == 1 {
            i += 1;
            continue;
        }

        map.entry(token.clone())
            .and_modify(|(count, indices)| {
                *count += 1;
                indices.push(i);
            })
            .or_insert((1, vec![i]));

        token.clear();
    }

    let (x, (_, _)) = match map.iter().max_by_key(|(_, (count, _))| count) {
        None => panic!("Error finding max"),
        Some((x, value)) => (x, value),
    };

    (x.clone(), map)
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
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut broken_words: Vec<Vec<String>> = Vec::new();
    for i in &words {
        let mut chars: Vec<String> = (*i).chars().map(String::from).collect();
        chars.insert(0, " ".to_string());
        broken_words.push(chars);
    }
    for i in vocabulary {
        for j in 0..broken_words.len() {
            for k in 0..(j - 1) {
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
    let mut returnable: Vec<String> = Vec::new();
    for i in 0..broken_words.len() {
        for j in 0..broken_words[i].len() {
            returnable.push(broken_words[i][j].clone());
        }
    }
    returnable
}
// TODO(LiuAlexR) - implement function to tokenize future inputs
