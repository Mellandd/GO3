use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use crate::go_loader::TermCounter;
use crate::go_ontology::{deepest_common_ancestor, get_term_by_id, get_terms_or_error, get_gene2go_or_error};

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


#[pyfunction]
pub fn compare_genes(
    gene1: &str,
    gene2: &str,
    ontology: String,
    similarity: String,
    groupwise: String,
    counter: &TermCounter,
) -> PyResult<f64> {
    let terms = get_terms_or_error()?;
    let gene2go = get_gene2go_or_error()?;
    let g1_terms = gene2go.get(gene1).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("Gene '{}' not found in mapping", gene1))
    })?;
    let g2_terms = gene2go.get(gene2).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("Gene '{}' not found in mapping", gene2))
    })?;
    let ns = match ontology.as_str() {
        "BP" => "biological_process",
        "MF" => "molecular_function",
        "CC" => "cellular_component",
        _ => {
            return Err(PyValueError::new_err(format!(
                "Invalid ontology '{}'. Must be 'BP', 'MF', or 'CC'",
                ontology
            )))
        }
    };
    let f1: Vec<String> = g1_terms
        .iter()
        .filter(|id| terms.get(*id).map_or(false, |t| t.namespace.to_ascii_lowercase() == ns))
        .cloned()
        .collect();

    let f2: Vec<String> = g2_terms
        .iter()
        .filter(|id| terms.get(*id).map_or(false, |t| t.namespace.to_ascii_lowercase() == ns))
        .cloned()
        .collect();
    print!("{:?}", f1);
    print!("{:?}", f2);
    if f1.is_empty() || f2.is_empty() {
        return Ok(0.0);
    }

    let sim_fn: fn(&str, &str, &TermCounter) -> f64 = match similarity.as_str() {
        "resnik" => resnik_similarity,
        "lin" => lin_similarity,
        _ => return Err(pyo3::exceptions::PyValueError::new_err("Unknown similarity")),
    };

    let score = match groupwise.as_str() {
        "max" => {
            f1.iter()
                .flat_map(|id1| f2.iter().map(move |id2| sim_fn(id1, id2, counter)))
                .fold(0.0, f64::max)
        }
        "bma" => {
            let sem1: Vec<f64> = f1.iter()
                .map(|id1| {
                    f2.iter()
                        .map(|id2| sim_fn(id1, id2, counter))
                        .fold(0.0, f64::max)
                })
                .collect();

            let sem2: Vec<f64> = f2.iter()
                .map(|id2| {
                    f1.iter()
                        .map(|id1| sim_fn(id1, id2, counter))
                        .fold(0.0, f64::max)
                })
                .collect();

            let total = sem1.len() + sem2.len();
            if total == 0 {
                0.0
            } else {
                (sem1.iter().sum::<f64>() + sem2.iter().sum::<f64>()) / total as f64
            }
        }
        _ => return Err(pyo3::exceptions::PyValueError::new_err("Unknown groupwise strategy")),
    };

    Ok(score)
}

#[pyfunction]
#[pyo3(signature = (pairs, ontology, method, combine, counter))]
pub fn compare_gene_pairs_batch(
    pairs: Vec<(String, String)>,
    ontology: String,
    method: String,
    combine: String,
    counter: &TermCounter,
) -> PyResult<Vec<f64>> {
    let gene2go = get_gene2go_or_error()?;
    let terms = get_terms_or_error()?;

    let ns = match ontology.as_str() {
        "BP" => "biological_process",
        "MF" => "molecular_function",
        "CC" => "cellular_component",
        _ => {
            return Err(PyValueError::new_err(format!(
                "Invalid ontology '{}'. Must be 'BP', 'MF', or 'CC'",
                ontology
            )))
        }
    };

    let sim_fn = match method.as_str() {
        "resnik" => resnik_similarity,
        "lin" => lin_similarity,
        _ => return Ok(vec![0.0; pairs.len()]), // más rápido que return en cada iter
    };

    let scores: Vec<f64> = pairs
        .into_par_iter()
        .map(|(g1, g2)| {
            let go1: Vec<_> = gene2go
                .get(&g1)
                .into_iter()
                .flatten()
                .filter(|go| terms.get(go.as_str()).map_or(false, |t| t.namespace.eq_ignore_ascii_case(ns)))
                .cloned()
                .collect();

            let go2: Vec<_> = gene2go
                .get(&g2)
                .into_iter()
                .flatten()
                .filter(|go| terms.get(go.as_str()).map_or(false, |t| t.namespace.eq_ignore_ascii_case(ns)))
                .cloned()
                .collect();

            if go1.is_empty() || go2.is_empty() {
                return 0.0;
            }

            match combine.as_str() {
                "max" => go1.iter()
                    .flat_map(|id1| go2.iter().map(move |id2| sim_fn(id1, id2, counter)))
                    .fold(0.0, f64::max),

                "bma" => {
                    let sem1: Vec<_> = go1.par_iter()
                        .map(|id1| {
                            go2.iter()
                                .map(|id2| sim_fn(id1, id2, counter))
                                .fold(0.0, f64::max)
                        })
                        .collect();

                    let sem2: Vec<_> = go2.par_iter()
                        .map(|id2| {
                            go1.iter()
                                .map(|id1| sim_fn(id1, id2, counter))
                                .fold(0.0, f64::max)
                        })
                        .collect();

                    let total = sem1.len() + sem2.len();
                    if total == 0 {
                        0.0
                    } else {
                        (sem1.iter().sum::<f64>() + sem2.iter().sum::<f64>()) / total as f64
                    }
                }

                _ => 0.0,
            }
        })
        .collect();

    Ok(scores)
}

// #[pyfunction]
// #[pyo3(signature = (pairs, ontology, method, combine, counter))]
// pub fn compare_gene_pairs_batch(
//     pairs: Vec<(String, String)>,
//     ontology: String,
//     method: String,
//     combine: String,
//     counter: &TermCounter,
// ) -> PyResult<Vec<f64>> {
//     let gene2go = get_gene2go_or_error()?;
//     let terms = get_terms_or_error()?;
//     let ns = match ontology.as_str() {
//         "BP" => "biological_process",
//         "MF" => "molecular_function",
//         "CC" => "cellular_component",
//         _ => {
//             return Err(PyValueError::new_err(format!(
//                 "Invalid ontology '{}'. Must be 'BP', 'MF', or 'CC'",
//                 ontology
//             )))
//         }
//     };
//     let scores: Vec<f64> = pairs
//         .into_iter()
//         .map(|(g1, g2)| {
//             let set1 = gene2go.get(&g1).cloned().unwrap_or_default();
//             let set2 = gene2go.get(&g2).cloned().unwrap_or_default();

//             let go1: Vec<String> = set1
//                 .into_iter()
//                 .filter(|go| terms.get(go).map_or(false, |t| t.namespace.to_ascii_lowercase() == ns))
//                 .collect();

//             let go2: Vec<String> = set2
//                 .into_iter()
//                 .filter(|go| terms.get(go).map_or(false, |t| t.namespace.to_ascii_lowercase() == ns))
//                 .collect();

//             if go1.is_empty() || go2.is_empty() {
//                 return 0.0;
//             }

//             let sim_fn = match method.as_str() {
//                 "resnik" => resnik_similarity,
//                 "lin" => lin_similarity,
//                 _ => return 0.0,
//             };

//             let mut matrix = vec![vec![0.0; go2.len()]; go1.len()];
//             for (i, id1) in go1.iter().enumerate() {
//                 for (j, id2) in go2.iter().enumerate() {
//                     matrix[i][j] = sim_fn(id1, id2, counter);
//                 }
//             }

//             match combine.as_str() {
//                 "max" => {
//                     go1.iter()
//                         .flat_map(|id1| go2.iter().map(move |id2| sim_fn(id1, id2, counter)))
//                         .fold(0.0, f64::max)
//                 }
//                 "bma" => {
//                     let sem1: Vec<f64> = go1.iter()
//                         .map(|id1| {
//                             go2.iter()
//                                 .map(|id2| sim_fn(id1, id2, counter))
//                                 .fold(0.0, f64::max)
//                         })
//                         .collect();

//                     let sem2: Vec<f64> = go2.iter()
//                         .map(|id2| {
//                             go1.iter()
//                                 .map(|id1| sim_fn(id1, id2, counter))
//                                 .fold(0.0, f64::max)
//                         })
//                         .collect();

//                     let total = sem1.len() + sem2.len();
//                     if total == 0 {
//                         0.0
//                     } else {
//                         (sem1.iter().sum::<f64>() + sem2.iter().sum::<f64>()) / total as f64
//                     }
//                 }
//                 _ => 0.0,
//             }
//         })
//         .collect();

//     Ok(scores)
// }