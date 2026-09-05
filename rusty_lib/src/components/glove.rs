use crate::{
    components::tokenizer::co_occurence,
    math::{Backend, DIMENSIONS, VOCAB},
};
use burn::Tensor;

const XMAX: i64 = 100;
const ALPHA: f64 = 0.75;

pub fn get_glove_embeddings(
    input: &[String],
    vocab: &[String],
    context_window: usize,
) -> Tensor<Backend, 2> {
    let device = Default::default();
    let shape = [VOCAB, DIMENSIONS];
    let embedding_matrix: Tensor<Backend, 2> = Tensor::<Backend, 2>::zeros(shape, &device);
    let co = co_occurence(input, vocab, context_window);

    embedding_matrix
}

// TODO(TheSilentIce)
// Create a co-occurence matrix
// Define context window
//
