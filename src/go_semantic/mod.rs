mod embedding;
mod gene;
mod similarity;
mod termset;

pub use embedding::{plot_embedding, plot_tsne_genes, plot_umap_genes, tsne_genes, umap_genes};
pub use gene::{
    compare_gene_pairs_batch, compare_gene_set_pairs_batch, compare_gene_set_profiles,
    compare_gene_sets, compare_genes, gene_distance_matrix, gene_set_distance_matrix,
    go_enrichment, EnrichmentResult,
};
pub use similarity::{
    batch_similarity, cluster_go_terms, reduce_redundant_terms, semantic_similarity,
    set_num_threads, term_distance_matrix, term_ic, term_similarity_matrix,
};
pub use termset::termset_similarity;

pub(crate) fn clear_internal_caches() {
    similarity::clear_internal_caches();
}
