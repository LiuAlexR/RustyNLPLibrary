use std::{cmp::Reverse, collections::HashMap};

use crate::{
    components::neural_net::{forward_pass, use_gelu},
    math::{create_random_matrix, create_random_vector, Backend},
};
use burn::{
    nn::loss::CrossEntropyLossConfig,
    tensor::{activation::softmax, cast::ToElement, Bool, Int, TensorData},
    Tensor,
};
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;

const DK: i64 = 64;
const DV: i64 = 64;
const D: i64 = 512;
const BLOCKS: i64 = 2;
const HEADS: i64 = 4;
const EPSILON: f64 = 1e-8;
const P_THRESHOLD: f64 = 0.9;

// Implement method to create final input matrix of [Nxd]
// Then method to calculate attention
//  make sure to save weights in a vector or somewhere,
//  remember, Q,K,V weights for each head, in each block
//  If there are 5 heads and 4 blocks, then there are a total of
//  3 x 5 x 4 = 60 weight matrices
// Implement LayerNorm

pub struct Transformer {
    blocks: Vec<Block>,
    num_heads: usize,
    d: usize,
    E: Tensor<Backend, 2>,
    gamma: Tensor<Backend, 1>,
    beta: Tensor<Backend, 1>,
    pad_token: String,
    pad_id: usize,
}

pub struct Block {
    gamma_pre: Tensor<Backend, 1>,  // [1xd]
    beta_pre: Tensor<Backend, 1>,   // [1xd]
    gamma_post: Tensor<Backend, 1>, // [1xd]
    beta_post: Tensor<Backend, 1>,  // [1xd]
    wq: Tensor<Backend, 2>,         // [dxd]
    wk: Tensor<Backend, 2>,         // [dxd]
    wv: Tensor<Backend, 2>,         // [dxd]
    wo: Tensor<Backend, 2>,         // [dxd]
    w_ffn_h: Tensor<Backend, 2>,    // [dxff]
    w_ffn_o: Tensor<Backend, 2>,    // [ffxd]
}

// implementing nucleus sampling
pub fn predict(
    prompt: &[String],
    context_window: usize,
    map: &HashMap<String, i64>,
    model: &Transformer,
    vocab: &[String],
    num_tokens: usize,
) -> String {
    let mut tokens = prompt.to_vec();

    for _ in 0..num_tokens {
        let start = tokens.len().saturating_sub(context_window);
        let context = &tokens[start..];

        let (b, _) = create_batches(
            context,
            context_window,
            1,
            model.E.clone(),
            map,
            &model.pad_token,
            model.d,
        );

        let o = softmax(transformer_forward_pass(b[0].clone(), model), 2);
        let [_, _, vocab_size] = o.dims();
        let actual_len = context.len();

        let p = o
            .slice([0..1, actual_len - 1..actual_len, 0..vocab_size])
            .reshape([vocab_size]);
        let next = predict_token(p, vocab);
        tokens.push(next);
    }

    tokens.join(" ")
}

pub fn predict_token(p: Tensor<Backend, 1>, vocab: &[String]) -> String {
    let mut probs: Vec<(f64, usize)> = p
        .into_data()
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .enumerate()
        .map(|(idx, prob)| (prob as f64, idx))
        .collect();

    probs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut c = 0.;
    let mut i = 0;
    let mut v = vec![];

    while c <= P_THRESHOLD && i < probs.len() {
        if vocab[probs[i].1] == "<PADD>" {
            i += 1;
            continue;
        }

        c += probs[i].0;
        v.push(probs[i]);
        i += 1;
    }

    let a: Vec<(f64, usize)> = v.into_iter().map(|(prob, idx)| (prob / c, idx)).collect();

    let weights: Vec<f64> = a.iter().map(|(prob, _)| *prob).collect();

    let dist = WeightedIndex::new(&weights).unwrap();

    let mut rng = rand::rng();
    let chosen = a[dist.sample(&mut rng)];
    let id = chosen.1;

    vocab[id].clone()
}

pub fn train(
    tokens: &[String],
    model: &mut Transformer,
    context_window: usize,
    batch_size: usize,
    map: &HashMap<String, i64>,
    learning_rate: f64,
) {
    let (v, t) = create_batches(
        tokens,
        context_window,
        batch_size,
        model.E.clone(),
        map,
        &model.pad_token,
        model.d,
    );

    for (x, y) in v.into_iter().zip(t.into_iter()) {
        let o = transformer_forward_pass(x, model);
        transformer_backward_pass(o, y, model, learning_rate);
    }
}

pub fn init_transformer(
    num_blocks: usize,
    num_heads: usize,
    d: usize,
    ff: usize,
    E: Tensor<Backend, 2>,
    pad_token: String,
    pad_id: usize,
) -> Transformer {
    let blocks = init_weights(num_blocks, d, ff);
    let gamma = Tensor::<Backend, 1>::ones([d], &Default::default()).require_grad();
    let beta = Tensor::<Backend, 1>::zeros([d], &Default::default()).require_grad();
    Transformer {
        blocks,
        num_heads,
        d,
        E,
        gamma,
        beta,
        pad_token,
        pad_id,
    }
}

fn transformer_backward_pass(
    logits: Tensor<Backend, 3>,
    targets: Tensor<Backend, 2, Int>,
    model: &mut Transformer,
    lr: f64,
) {
    let [batch, len, vocab] = logits.dims();
    let logits = logits.reshape([batch * len, vocab]);
    let targets = targets.reshape([batch * len]);

    let loss = CrossEntropyLossConfig::new()
        .with_pad_tokens(Some(vec![model.pad_id]))
        .init(&logits.device())
        .forward(logits, targets);

    // println!("loss: {}", loss.clone().into_scalar());

    let grads = loss.backward();

    macro_rules! update {
        ($field:expr) => {
            let g = $field.grad(&grads).unwrap();
            let updated = $field.clone().inner() - g.mul_scalar(lr);
            $field = Tensor::from_inner(updated).require_grad();
        };
    }

    update!(model.E);
    update!(model.gamma);
    update!(model.beta);

    for block in model.blocks.iter_mut() {
        update!(block.gamma_pre);
        update!(block.beta_pre);
        update!(block.gamma_post);
        update!(block.beta_post);
        update!(block.wq);
        update!(block.wk);
        update!(block.wv);
        update!(block.wo);
        update!(block.w_ffn_h);
        update!(block.w_ffn_o);
    }
}

fn transformer_forward_pass(mut X: Tensor<Backend, 3>, model: &Transformer) -> Tensor<Backend, 3> {
    for block in &model.blocks {
        X = run_block(X, block, model.d, model.num_heads);
    }

    let X = layer_norm(X, model.gamma.clone(), model.beta.clone());
    X.matmul(model.E.clone().transpose().unsqueeze())
}

fn run_block(X: Tensor<Backend, 3>, b: &Block, d: usize, heads: usize) -> Tensor<Backend, 3> {
    let l = layer_norm(X.clone(), b.gamma_pre.clone(), b.beta_pre.clone());
    let A = X.clone().add(calculate_attention(l, heads, d, b));

    let l = layer_norm(A.clone(), b.gamma_post.clone(), b.beta_post.clone());

    A.clone().add(forward_pass(
        l,
        b.w_ffn_h.clone(),
        b.w_ffn_o.clone(),
        use_gelu,
    ))
}

/// input shape : batch x len x dimension
fn calculate_attention(
    X: Tensor<Backend, 3>,
    num_heads: usize,
    d: usize,
    b: &Block,
) -> Tensor<Backend, 3> {
    // shapes are still batch x len x d
    let Q = X.clone().matmul(b.wq.clone().unsqueeze());
    let K = X.clone().matmul(b.wk.clone().unsqueeze());
    let V = X.clone().matmul(b.wv.clone().unsqueeze());

    let batch_size = Q.dims()[0];
    let len = Q.dims()[1];

    // shapes are now batch x len x num_heads x (d / num_heads)
    let Q: Tensor<Backend, 4> = Q.reshape([batch_size, len, num_heads, d / num_heads]);
    let K: Tensor<Backend, 4> = K.reshape([batch_size, len, num_heads, d / num_heads]);
    let V: Tensor<Backend, 4> = V.reshape([batch_size, len, num_heads, d / num_heads]);

    // shapes are now batch x num_heads x len x (d / num_heads)
    let Q = Q.swap_dims(1, 2);
    let K = K.swap_dims(1, 2);
    let V = V.swap_dims(1, 2);

    // transposing K along the last 2 dims
    // K is now of shape batch x num_heads (d / num_heads) x len
    let K = K.swap_dims(2, 3);

    // product shape is now batch x heads x len x len
    let head_dim = d / num_heads;
    // let p = Q.matmul(K).div_scalar((d as f64).sqrt());
    let p = Q.matmul(K).div_scalar((head_dim as f64).sqrt());

    let mask = Tensor::<Backend, 4, Bool>::tril_mask(p.dims(), 0, &Default::default());
    let mask = mask.bool_not();
    let p = p.mask_fill(mask, f32::MIN);

    // softmaxing over the last dim, which represents the keys
    let p = softmax(p, 3);

    // shape is now batch x heads x len x d_h
    let p = p.matmul(V);

    // transposing back
    // shape is now batch x len x heads x d_h
    let p = p.swap_dims(1, 2);

    // reshaping into shape batch x len x d
    let p = p.reshape([batch_size, len, d]);

    p.matmul(b.wo.clone().unsqueeze())
}

/// Applies LayerNorm to input
fn layer_norm(
    X: Tensor<Backend, 3>,
    gamma: Tensor<Backend, 1>,
    beta: Tensor<Backend, 1>,
) -> Tensor<Backend, 3> {
    let mean = X.clone().mean_dim(2);
    let variance = X.clone().sub(mean.clone()).powf_scalar(2.).mean_dim(2);

    let normalized = X.sub(mean).div(variance.add_scalar(EPSILON).sqrt());

    normalized.mul(gamma.unsqueeze()).add(beta.unsqueeze())
}
/// initializes weights of all heads in all blocks
///
/// ith block in vec is indicative of a block
fn init_weights(blocks: usize, d: usize, ff: usize) -> Vec<Block> {
    let mut res: Vec<Block> = Vec::with_capacity(blocks as usize);

    for _ in 0..blocks {
        res.push(Block {
            gamma_pre: Tensor::<Backend, 1>::ones([d], &Default::default()).require_grad(),
            beta_pre: Tensor::<Backend, 1>::zeros([d], &Default::default()).require_grad(),
            gamma_post: Tensor::<Backend, 1>::ones([d], &Default::default()).require_grad(),
            beta_post: Tensor::<Backend, 1>::zeros([d], &Default::default()).require_grad(),
            wq: create_random_matrix(d, d).require_grad(),
            wk: create_random_matrix(d, d).require_grad(),
            wv: create_random_matrix(d, d).require_grad(),
            wo: create_random_matrix(d, d).require_grad(),
            w_ffn_h: create_random_matrix(d, ff).require_grad(),
            w_ffn_o: create_random_matrix(ff, d).require_grad(),
        });
    }
    res
}

/// Creates combined embeddings from word and positional embeddings
///
/// Returns a 2D Tensor of shape [Nxd] where `N` is number of tokens and
/// `d` is modal dimensionality
pub fn generate_combined_embeddings(
    tokens: &[String],
    map: &HashMap<String, i64>,
    embedding_matrix: Tensor<Backend, 2>,
    positional_embeddings: &Tensor<Backend, 2>,
) -> Tensor<Backend, 2> {
    let ids: Vec<i64> = tokens.iter().map(|t| map[t]).collect();

    let device = embedding_matrix.device();
    let indices =
        Tensor::<Backend, 1, Int>::from_data(TensorData::new(ids, [tokens.len()]), &device);

    let words = embedding_matrix.select(0, indices);
    words.add(positional_embeddings.clone())
}

/// Generates sinusoidal position embeddings for `length` tokens of dimensionality `d`
///
/// Returns a 2D Tensor where each row vector `i` is the positional embedding for the `ith` token
fn generate_positional_embeddings(length: usize, d: usize) -> Tensor<Backend, 2> {
    let mut res: Vec<Vec<f64>> = vec![vec![]; length];

    for pos in 0..length {
        let pos = pos as f64;
        let mut v: Vec<f64> = vec![0.; d];

        for i in 0..d / 2 {
            let denominator = 10000_f64.powf(2. * i as f64 / d as f64);
            v[2 * i] = (pos / denominator).sin();
            v[2 * i + 1] = (pos / denominator).cos();
        }
        res[pos as usize] = v.clone();
    }

    let shape = [length, d];
    let t = TensorData::new(res.into_iter().flatten().collect(), shape);
    Tensor::<Backend, 2>::from_data(t, &Default::default())
}
fn create_batches(
    tokens: &[String],
    context_window: usize,
    batch_size: usize,
    embedding_matrix: Tensor<Backend, 2>,
    map: &HashMap<String, i64>,
    pad_token: &str,
    d: usize,
) -> (Vec<Tensor<Backend, 3>>, Vec<Tensor<Backend, 2, Int>>) {
    let device = embedding_matrix.device();
    let n = tokens.len();

    // computed once — every chunk is padded to context_window, so this is reused as-is
    let pos_emb = generate_positional_embeddings(context_window, d);

    let mut sequences: Vec<Tensor<Backend, 2>> = Vec::new();
    let mut target_sequences: Vec<Tensor<Backend, 1, Int>> = Vec::new();

    let mut start = 0;
    while start < n {
        let end = (start + context_window).min(n);

        let mut chunk_tokens = tokens[start..end].to_vec();
        while chunk_tokens.len() < context_window {
            chunk_tokens.push(pad_token.to_string());
        }

        let target_start = (start + 1).min(n);
        let target_end = (target_start + context_window).min(n);
        let mut target_tokens = tokens[target_start..target_end].to_vec();
        while target_tokens.len() < context_window {
            target_tokens.push(pad_token.to_string());
        }

        let embedded =
            generate_combined_embeddings(&chunk_tokens, map, embedding_matrix.clone(), &pos_emb);
        sequences.push(embedded);

        let target_ids: Vec<i64> = target_tokens.iter().map(|t| map[t]).collect();
        let target_tensor = Tensor::<Backend, 1, Int>::from_data(
            TensorData::new(target_ids, [context_window]),
            &device,
        );
        target_sequences.push(target_tensor);

        start += context_window;
    }

    let input_batches = sequences
        .chunks(batch_size)
        .map(|group| Tensor::stack::<3>(group.to_vec(), 0))
        .collect();

    let target_batches = target_sequences
        .chunks(batch_size)
        .map(|group| Tensor::stack::<2>(group.to_vec(), 0))
        .collect();

    (input_batches, target_batches)
}
