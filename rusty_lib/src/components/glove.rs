use crate::math::{update_matrix, Backend, DIMENSIONS, VOCAB};
use burn::Tensor;
use std::collections::HashMap;

const ALPHA: f32 = 0.75;
const XMAX: u64 = 100;
const CONTEXT_WINDOW: u64 = 10;

pub fn get_glove_embeddings(
    input: &[String],
    vocab: &[String],
    context_window: usize,
) -> Tensor<Backend, 2> {
    let device = Default::default();
    let shape = [VOCAB, DIMENSIONS];
    let embedding_matrix: Tensor<Backend, 2> = Tensor::<Backend, 2>::zeros(shape, &device);

    let matrix = co_occurence(input, vocab, context_window);

    embedding_matrix
}

fn count_word(
    target_idx: usize,
    map: &mut HashMap<&str, (usize, Vec<u64>)>,
    context_window: usize,
    input: &[String],
) {
    let s: &str = &input[target_idx];
    let start = target_idx.saturating_sub(context_window);
    let end = (target_idx + context_window).min(input.len() - 1);

    for i in start..=end {
        if i == target_idx {
            continue;
        }

        let context_idx = map.get(&input[i] as &str).unwrap().0;

        if let Some((_, counts)) = map.get_mut(s) {
            counts[context_idx] += 1;
        }
    }
}

// creates co_occurence matrix for a given input
fn co_occurence(input: &[String], vocab: &[String], context_window: usize) -> Tensor<Backend, 2> {
    assert!(context_window >= 1, "Context window must be at least 1");
    assert_eq!(
        vocab.len(),
        VOCAB + 128,
        "vocab length must match VOCAB constant"
    );

    let device = Default::default();
    let mut matrix = Tensor::<Backend, 2>::zeros([vocab.len(), vocab.len()], &device);

    // map each token to its corresponding index in vocab
    let mut map: HashMap<&str, (usize, Vec<u64>)> = vocab
        .iter()
        .enumerate()
        .map(|(idx, s)| (s.as_str(), (idx, vec![0u64; vocab.len()])))
        .collect();

    for idx in 0..input.len() {
        count_word(idx, &mut map, context_window, input);
    }

    for (_, (idx, counts)) in map.iter() {
        let counts_f32: Vec<f32> = counts.iter().map(|&c| c as f32).collect();
        let row: Tensor<Backend, 1> = Tensor::from_data(counts_f32.as_slice(), &device);
        let row_2d: Tensor<Backend, 2> = row.unsqueeze();
        matrix = update_matrix(matrix, row_2d, idx + 1);
    }

    matrix
}

// TODO(TheSilentIce)
// Create a co-occurence matrix
// Define context window
