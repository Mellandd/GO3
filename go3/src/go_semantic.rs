use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use crate::go_loader::TermCounter;
use crate::go_ontology::{deepest_common_ancestor, get_term_by_id};

/// Compute Information Content (IC) for the given GO term using the annotations.
///
/// # Arguments
///
/// * `go_id` - GO term ID.
/// * `counter` - TermCounter with the annotations.
///
/// # Returns
///
/// Information Content (float)
#[pyfunction]
pub fn term_ic(go_id: &str, counter: &TermCounter) -> f64 {
    *counter.ic.get(go_id).unwrap_or(&0.0)
}

/// Compute similarity between two GO terms using Resnik.
///
/// # Arguments
///
/// * `id1` - First GO term ID
/// * `id2` - Second GO term ID
///
/// # Returns
///
/// Resnik similarity score (float)
#[pyfunction]
pub fn resnik_similarity(id1: &str, id2: &str, counter: &TermCounter) -> f64 {
    let (t1, t2) = match (get_term_by_id(id1).ok().flatten(), get_term_by_id(id2).ok().flatten()) {
        (Some(t1), Some(t2)) => (t1, t2),
        _ => return 0.0,
    };

    if t1.namespace != t2.namespace {
        return 0.0;
    }

    match deepest_common_ancestor(id1, id2).ok().flatten() {
        Some(dca) => term_ic(&dca, counter),
        None => 0.0,
    }
}

/// Compute similarity between two GO terms using Lin.
///
/// # Arguments
///
/// * `id1` - First GO term ID
/// * `id2` - Second GO term ID
///
/// # Returns
///
/// Lin similarity score (float)
#[pyfunction]
pub fn lin_similarity(id1: &str, id2: &str, counter: &TermCounter) -> f64 {
    let resnik = resnik_similarity(id1, id2, counter);
    if resnik == 0.0 {
        return 0.0;
    }

    let (ic1, ic2) = (term_ic(id1, counter), term_ic(id2, counter));
    if ic1 == 0.0 || ic2 == 0.0 {
        return 0.0;
    }

    2.0 * resnik / (ic1 + ic2)
}

/// Compute similarity between two batches of GO terms using Resnik similarity.
/// Both lists must be of the same size.
///
/// # Arguments
///
/// * `list1` - First list of GO term ID
/// * `list2` - Second list GO term ID
/// * `counter` - TermCounter with the annotations.
/// # Returns
///
/// List of Resnik similarity scores (float)
#[pyfunction]
pub fn batch_resnik(list1: Vec<String>, list2: Vec<String>, counter: &TermCounter) -> PyResult<Vec<f64>> {
    if list1.len() != list2.len() {
        return Err(PyValueError::new_err("Both lists must be the same length"));
    }

    Ok(list1
        .par_iter()
        .zip(list2.par_iter())
        .map(|(id1, id2)| {
            match deepest_common_ancestor(id1, id2) {
                Ok(Some(dca)) => *counter.ic.get(&dca).unwrap_or(&0.0),
                _ => 0.0,
            }
        })
        .collect())
}

/// Compute similarity between two batches of GO terms using Resnik similarity.
/// Both lists must be of the same size.
///
/// # Arguments
///
/// * `list1` - First list of GO term ID
/// * `list2` - Second list GO term ID
/// * `counter` - TermCounter with the annotations.
/// # Returns
///
/// List of Resnik similarity scores (float)
#[pyfunction]
pub fn batch_lin(list1: Vec<String>, list2: Vec<String>, counter: &TermCounter) -> PyResult<Vec<f64>> {
    if list1.len() != list2.len() {
        return Err(PyValueError::new_err("Both lists must be the same length"));
    }

    Ok(list1
        .par_iter()
        .zip(list2.par_iter())
        .map(|(id1, id2)| {
            let resnik = match deepest_common_ancestor(id1, id2) {
                Ok(Some(dca)) => *counter.ic.get(&dca).unwrap_or(&0.0),
                _ => return 0.0,
            };

            if resnik == 0.0 {
                return 0.0;
            }

            let ic1 = *counter.ic.get(id1).unwrap_or(&0.0);
            let ic2 = *counter.ic.get(id2).unwrap_or(&0.0);
            if ic1 == 0.0 || ic2 == 0.0 {
                return 0.0;
            }

            2.0 * resnik / (ic1 + ic2)
        })
        .collect())
}