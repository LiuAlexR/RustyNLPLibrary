use std::{collections::HashMap, time::Instant, vec::Vec};

/// Byte-Pair Encoding scheme
/// @param corpus - text to tokenize
/// @param num_tokens - maximum number of tokens in vocabulary plus default ASCII
pub fn bpe_tokenize(corpus: &str, num_tokens: u64, only_new: bool) -> Vec<String> {
    let mut vocabulary: Vec<String> = (0..128).map(|b: u8| (b as char).to_string()).collect();

    let mut new_vocab: Vec<String> = Vec::default();

    let arr: Vec<char> = corpus.chars().collect();

    let (token, mut map) = combine(&arr, &vocabulary);

    if only_new {
        new_vocab.push(token.clone());
    }
    vocabulary.push(token);

    for _ in 1..num_tokens {
        let x = Instant::now();
        let token = combine_with_index(&arr, &vocabulary, &mut map);
        println!("Time: {}μs", x.elapsed().as_micros());

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

// HashMap<String, tuple<u64, Vector<u64>
// Stores the indices of tokens
// so say we have <MA, <10, [3,20,30]>>
//
// then we skip to those indices, account for EOF and UNK
// characters, combine with the other token, insert and update

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

fn combine(arr: &[char], vocabulary: &[String]) -> (String, HashMap<String, (u64, Vec<u64>)>) {
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

    // for ties, returns the last one
    // not deterministic, as pointers are stored anywhere

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

// TODO(LiuAlexR) - implement function to tokenize future inputs
