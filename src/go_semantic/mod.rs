mod embedding;
mod gene;
mod similarity;
mod termset;

pub use embedding::{plot_embedding, plot_tsne_genes, plot_umap_genes, tsne_genes, umap_genes};
pub use gene::{
    compare_gene_pairs_batch, compare_gene_set_pairs_batch, compare_gene_set_profiles,
    compare_gene_sets, compare_genes, gene_distance_matrix, gene_set_distance_matrix,
};
pub use similarity::{batch_similarity, semantic_similarity, set_num_threads, term_ic};
pub use termset::termset_similarity;

pub(crate) fn clear_internal_caches() {
    similarity::clear_internal_caches();
}
