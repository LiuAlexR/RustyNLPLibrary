use crate::math::{Backend, DIMENSIONS, VOCAB};
use burn::Tensor;

pub fn get_glove_embeddings() -> Tensor<Backend, 2> {
    let device = Default::default();
    let shape = [VOCAB, DIMENSIONS];
    let embedding_matrix: Tensor<Backend, 2> = Tensor::<Backend, 2>::zeros(shape, &device);

    embedding_matrix
}

// TODO(TheSilentIce)
// Create a co-occurence matrix
// Define context window
