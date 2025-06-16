use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;

pub mod go_loader;
pub mod go_ontology;
pub mod go_semantic; // si ya tienes esta parte

use go_loader::{load_go_terms, load_gaf, build_term_counter};
use go_ontology::{get_term_by_id, ancestors, common_ancestor, deepest_common_ancestor, inspect_gene2go_cache};
use go_semantic::{term_ic, resnik_similarity, lin_similarity, batch_resnik, batch_lin, compare_genes, compare_gene_pairs_batch}; // si ya están

#[pymodule]
fn go3(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_go_terms, m)?)?;
    m.add_function(wrap_pyfunction!(load_gaf, m)?)?;
    m.add_function(wrap_pyfunction!(build_term_counter, m)?)?;

    m.add_function(wrap_pyfunction!(get_term_by_id, m)?)?;
    m.add_function(wrap_pyfunction!(ancestors, m)?)?;
    m.add_function(wrap_pyfunction!(common_ancestor, m)?)?;
    m.add_function(wrap_pyfunction!(deepest_common_ancestor, m)?)?;
    m.add_function(wrap_pyfunction!(inspect_gene2go_cache, m)?)?;

    m.add_function(wrap_pyfunction!(term_ic, m)?)?;
    m.add_function(wrap_pyfunction!(resnik_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(lin_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(batch_resnik, m)?)?;
    m.add_function(wrap_pyfunction!(batch_lin, m)?)?;
    m.add_function(wrap_pyfunction!(compare_genes, m)?)?;
    m.add_function(wrap_pyfunction!(compare_gene_pairs_batch, m)?)?;

    m.add_class::<go_ontology::PyGOTerm>()?;

    Ok(())
}