use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::components::tokenizer::bpe_encoder;

pub struct BM25Index {
    postings: HashMap<String, Vec<(usize, usize)>>, // term -> [(doc_id, term_freq)]
    doc_lengths: Vec<usize>,
    avg_doc_length: f64,
    num_docs: usize,
    doc_freq: HashMap<String, usize>, // term -> number of docs containing it
    doc_paths: Vec<String>,
    doc_texts: Vec<String>, // raw article text, for feeding back into the model post-retrieval
    k1: f64,
    b: f64,
}

impl BM25Index {
    /// Builds an index from every `.txt` file in `dir`
    pub fn build_from_directory(
        dir: &Path,
        vocab: &[String],
        k1: f64,
        b: f64,
    ) -> std::io::Result<Self> {
        let mut documents: Vec<Vec<String>> = Vec::new();
        let mut doc_paths: Vec<String> = Vec::new();
        let mut doc_texts: Vec<String> = Vec::new();

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("txt") {
                let text = fs::read_to_string(&path)?;
                let tokens = bpe_encoder(&vocab.to_vec(), &text);
                documents.push(tokens);
                doc_paths.push(path.to_string_lossy().to_string());
                doc_texts.push(text);
            }
        }

        Ok(Self::build(&documents, doc_paths, doc_texts, k1, b))
    }

    fn build(
        documents: &[Vec<String>],
        doc_paths: Vec<String>,
        doc_texts: Vec<String>,
        k1: f64,
        b: f64,
    ) -> Self {
        let num_docs = documents.len();
        let mut postings: HashMap<String, Vec<(usize, usize)>> = HashMap::new();
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let mut doc_lengths = Vec::with_capacity(num_docs);

        for (doc_id, doc) in documents.iter().enumerate() {
            doc_lengths.push(doc.len());

            let mut term_counts: HashMap<&str, usize> = HashMap::new();
            for token in doc {
                *term_counts.entry(token.as_str()).or_insert(0) += 1;
            }

            for (term, &count) in &term_counts {
                postings
                    .entry(term.to_string())
                    .or_default()
                    .push((doc_id, count));
                *doc_freq.entry(term.to_string()).or_insert(0) += 1;
            }
        }

        let avg_doc_length = doc_lengths.iter().sum::<usize>() as f64 / num_docs.max(1) as f64;

        BM25Index {
            postings,
            doc_lengths,
            avg_doc_length,
            num_docs,
            doc_freq,
            doc_paths,
            doc_texts,
            k1,
            b,
        }
    }

    /// Scores every document against a query (already BPE-tokenized), returns
    /// (file_path, score) pairs sorted descending, truncated to `top_k`.
    pub fn search(&self, query: &[String], top_k: usize) -> Vec<(String, f64)> {
        let mut scores: HashMap<usize, f64> = HashMap::new();

        for term in query {
            let Some(matching_docs) = self.postings.get(term) else {
                continue;
            };
            let idf = self.idf(term);

            for &(doc_id, term_freq) in matching_docs {
                let doc_len = self.doc_lengths[doc_id] as f64;
                let tf = term_freq as f64;

                let numerator = tf * (self.k1 + 1.0);
                let denominator =
                    tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_length);

                *scores.entry(doc_id).or_insert(0.0) += idf * numerator / denominator;
            }
        }

        let mut ranked: Vec<(usize, f64)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked.truncate(top_k);

        ranked
            .into_iter()
            .map(|(doc_id, score)| (self.doc_paths[doc_id].clone(), score))
            .collect()
    }

    /// Returns the raw text of a document given its file path (as returned by `search`).
    pub fn get_text(&self, path: &str) -> Option<&str> {
        self.doc_paths
            .iter()
            .position(|p| p == path)
            .map(|idx| self.doc_texts[idx].as_str())
    }

    fn idf(&self, term: &str) -> f64 {
        let n = self.num_docs as f64;
        let df = *self.doc_freq.get(term).unwrap_or(&0) as f64;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }
}
