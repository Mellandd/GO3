use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;

pub mod go_loader;
pub mod go_ontology;
pub mod go_semantic;

use go_loader::{build_term_counter, load_gaf, load_go_terms};
use go_ontology::{
    ancestors, common_ancestor, deepest_common_ancestor, descendants, get_genes_for_term,
    get_term_by_id, get_terms_for_gene, list_genes,
};
use go_semantic::{
    batch_similarity, cluster_go_terms, compare_gene_pairs_batch, compare_gene_set_pairs_batch,
    compare_gene_set_profiles, compare_gene_sets, compare_genes, gene_distance_matrix,
    gene_set_distance_matrix, go_enrichment, plot_embedding, plot_tsne_genes, plot_umap_genes,
    reduce_redundant_terms, semantic_similarity, set_num_threads, term_distance_matrix, term_ic,
    term_similarity_matrix, termset_similarity, tsne_genes, umap_genes, EnrichmentResult,
};

#[pymodule]
fn go3(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_go_terms, m)?)?;
    m.add_function(wrap_pyfunction!(load_gaf, m)?)?;
    m.add_function(wrap_pyfunction!(build_term_counter, m)?)?;

    m.add_function(wrap_pyfunction!(get_term_by_id, m)?)?;
    m.add_function(wrap_pyfunction!(ancestors, m)?)?;
    m.add_function(wrap_pyfunction!(descendants, m)?)?;
    m.add_function(wrap_pyfunction!(common_ancestor, m)?)?;
    m.add_function(wrap_pyfunction!(deepest_common_ancestor, m)?)?;
    m.add_function(wrap_pyfunction!(list_genes, m)?)?;
    m.add_function(wrap_pyfunction!(get_terms_for_gene, m)?)?;
    m.add_function(wrap_pyfunction!(get_genes_for_term, m)?)?;

    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(term_ic, m)?)?;
    m.add_function(wrap_pyfunction!(semantic_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(term_similarity_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(term_distance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(reduce_redundant_terms, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_go_terms, m)?)?;
    m.add_function(wrap_pyfunction!(termset_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(batch_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(compare_genes, m)?)?;
    m.add_function(wrap_pyfunction!(compare_gene_pairs_batch, m)?)?;
    m.add_function(wrap_pyfunction!(compare_gene_sets, m)?)?;
    m.add_function(wrap_pyfunction!(compare_gene_set_pairs_batch, m)?)?;
    m.add_function(wrap_pyfunction!(compare_gene_set_profiles, m)?)?;
    m.add_function(wrap_pyfunction!(gene_distance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(gene_set_distance_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(go_enrichment, m)?)?;
    m.add_function(wrap_pyfunction!(tsne_genes, m)?)?;
    m.add_function(wrap_pyfunction!(umap_genes, m)?)?;
    m.add_function(wrap_pyfunction!(plot_embedding, m)?)?;
    m.add_function(wrap_pyfunction!(plot_tsne_genes, m)?)?;
    m.add_function(wrap_pyfunction!(plot_umap_genes, m)?)?;

    m.add_class::<go_ontology::PyGOTerm>()?;
    m.add_class::<go_loader::GAFAnnotation>()?;
    m.add_class::<go_loader::GAFParseOptions>()?;
    m.add_class::<go_loader::TermCounter>()?;
    m.add_class::<EnrichmentResult>()?;

    Ok(())
}
