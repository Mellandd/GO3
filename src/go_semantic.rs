use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use crate::go_loader::TermCounter;
use std::collections::{HashSet, HashMap};
use crate::go_ontology::{deepest_common_ancestor, get_term_by_id, get_terms_or_error, get_gene2go_or_error};

/// Compute the Information Content (IC) of a GO term.
#[pyfunction]
#[pyo3(text_signature = "(go_id, counter)")]
/// Returns the Information Content (IC) value of a given GO term.
///
/// Parameters:
///     go_id (str): GO term identifier.
///     counter (TermCounter): Precomputed term counter with IC values.
///
/// Returns:
///     float: The IC of the GO term.
pub fn term_ic(go_id: &str, counter: &TermCounter) -> f64 {
    *counter.ic.get(go_id).unwrap_or(&0.0)
}

/// Compute the Resnik similarity between two GO terms.
#[pyfunction]
#[pyo3(text_signature = "(id1, id2, counter)")]
/// Returns the Resnik similarity between two GO terms.
///
/// Parameters:
///     id1 (str): First GO term ID.
///     id2 (str): Second GO term ID.
///     counter (TermCounter): Precomputed term counter with IC values.
///
/// Returns:
///     float: Resnik similarity value.
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

/// Compute the Lin similarity between two GO terms.
#[pyfunction]
#[pyo3(text_signature = "(id1, id2, counter)")]
/// Returns the Lin similarity between two GO terms.
///
/// Parameters:
///     id1 (str): First GO term ID.
///     id2 (str): Second GO term ID.
///     counter (TermCounter): Precomputed term counter with IC values.
///
/// Returns:
///     float: Lin similarity value.
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

#[derive(Debug)]
enum SimilarityMethod {
    Resnik,
    Lin,
    JC,
    SimRel,
    ICCoef,
    GraphIC,
    Wang,
}

impl SimilarityMethod {
    fn from_str(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "resnik" => Some(SimilarityMethod::Resnik),
            "lin" => Some(SimilarityMethod::Lin),
            "jc" => Some(SimilarityMethod::JC),
            "simrel" => Some(SimilarityMethod::SimRel),
            "iccoef" => Some(SimilarityMethod::ICCoef),
            "graphic" => Some(SimilarityMethod::GraphIC),
            "wang" => Some(SimilarityMethod::Wang),
            _ => None,
        }
    }

    fn compute(&self, id1: &str, id2: &str, counter: &TermCounter) -> f64 {
        match self {
            SimilarityMethod::Resnik => {
                let dca = match deepest_common_ancestor(id1, id2).ok().flatten() {
                    Some(dca) => dca,
                    None => return 0.0,
                };
                *counter.ic.get(&dca).unwrap_or(&0.0)
            }
            SimilarityMethod::Lin => {
                let dca = match deepest_common_ancestor(id1, id2).ok().flatten() {
                    Some(dca) => dca,
                    None => return 0.0,
                };
                let resnik = *counter.ic.get(&dca).unwrap_or(&0.0);
                if resnik == 0.0 {
                    return 0.0;
                }
                let ic1 = *counter.ic.get(id1).unwrap_or(&0.0);
                let ic2 = *counter.ic.get(id2).unwrap_or(&0.0);
                if ic1 == 0.0 || ic2 == 0.0 {
                    return 0.0;
                }
                2.0 * resnik / (ic1 + ic2)
            }
            SimilarityMethod::JC => {
                let (t1, t2) = match (get_term_by_id(id1).ok().flatten(), get_term_by_id(id2).ok().flatten()) {
                    (Some(t1), Some(t2)) => (t1, t2),
                    _ => return 0.0,
                };
            
                if t1.namespace != t2.namespace {
                    return 0.0;
                }
            
                let ic1 = term_ic(id1, counter);
                let ic2 = term_ic(id2, counter);
            
                let dca_ic = match deepest_common_ancestor(id1, id2).ok().flatten() {
                    Some(dca) => term_ic(&dca, counter),
                    None => return 0.0,
                };
            
                let distance = ic1 + ic2 - 2.0 * dca_ic;
                if distance <= 0.0 {
                    return f64::INFINITY;  // Máxima similitud
                }
                distance
            }
            SimilarityMethod::SimRel => {
                let (t1, t2) = match (get_term_by_id(id1).ok().flatten(), get_term_by_id(id2).ok().flatten()) {
                    (Some(t1), Some(t2)) => (t1, t2),
                    _ => return 0.0,
                };
            
                if t1.namespace != t2.namespace {
                    return 0.0;
                }
            
                let ic1 = term_ic(id1, counter);
                let ic2 = term_ic(id2, counter);
            
                if ic1 == 0.0 || ic2 == 0.0 {
                    return 0.0;
                }
            
                let dca_ic = match deepest_common_ancestor(id1, id2).ok().flatten() {
                    Some(dca) => term_ic(&dca, counter),
                    None => return 0.0,
                };
            
                if dca_ic == 0.0 {
                    return 0.0;
                }
            
                let lin = (2.0 * dca_ic) / (ic1 + ic2);
                lin * (1.0 - (-dca_ic).exp())
            }
            SimilarityMethod::ICCoef => {
                let (t1, t2) = match (get_term_by_id(id1).ok().flatten(), get_term_by_id(id2).ok().flatten()) {
                    (Some(t1), Some(t2)) => (t1, t2),
                    _ => return 0.0,
                };
            
                if t1.namespace != t2.namespace {
                    return 0.0;
                }
            
                let ic1 = term_ic(id1, counter);
                let ic2 = term_ic(id2, counter);
            
                if ic1 == 0.0 || ic2 == 0.0 {
                    return 0.0;
                }
            
                let dca_ic = match deepest_common_ancestor(id1, id2).ok().flatten() {
                    Some(dca) => term_ic(&dca, counter),
                    None => return 0.0,
                };
            
                dca_ic / ic1.min(ic2)
            }
            SimilarityMethod::GraphIC => {
                let (t1, t2) = match (get_term_by_id(id1).ok().flatten(), get_term_by_id(id2).ok().flatten()) {
                    (Some(t1), Some(t2)) => (t1, t2),
                    _ => return 0.0,
                };
            
                if t1.namespace != t2.namespace {
                    return 0.0;
                }
            
                let depth1 = t1.depth.unwrap_or(0);
                let depth2 = t2.depth.unwrap_or(0);
                let max_depth = (depth1.max(depth2) + 1) as f64;
            
                let dca_ic = match deepest_common_ancestor(id1, id2).ok().flatten() {
                    Some(dca) => term_ic(&dca, counter),
                    None => return 0.0,
                };
            
                dca_ic / max_depth
            }
            SimilarityMethod::Wang => {
                let terms_guard = match crate::go_loader::GO_TERMS_CACHE.get() {
                    Some(lock) => match lock.read() {
                        Ok(terms) => terms,
                        Err(_) => return 0.0,
                    },
                    None => return 0.0,
                };
            
                let terms = &*terms_guard;
            
                let t1 = match terms.get(id1) {
                    Some(t) => t,
                    None => return 0.0,
                };
                let t2 = match terms.get(id2) {
                    Some(t) => t,
                    None => return 0.0,
                };
            
                if t1.namespace != t2.namespace {
                    return 0.0;
                }
            
                let sv_a = semantic_contributions(id1, terms);
                let sv_b = semantic_contributions(id2, terms);
            
                let sum_a: f64 = sv_a.values().sum();
                let sum_b: f64 = sv_b.values().sum();
            
                let common_keys: std::collections::HashSet<_> = sv_a.keys().collect::<HashSet<_>>()
                    .intersection(&sv_b.keys().collect::<HashSet<_>>())
                    .cloned()
                    .collect();
            
                let mut numerator = 0.0;
                for key in common_keys {
                    if let (Some(w1), Some(w2)) = (sv_a.get(key), sv_b.get(key)) {
                        numerator += (*w1).min(*w2);
                    }
                }
            
                if sum_a + sum_b == 0.0 {
                    0.0
                } else {
                    numerator / ((sum_a + sum_b) / 2.0)
                }
            }
        }
    }
}

/// Compute semantic similarity between two GO terms using a selected method.
///
/// Args:
///     id1 (str): First GO term ID.
///     id2 (str): Second GO term ID.
///     method (str): Name of the similarity method. Options: `"resnik"`, `"lin"`.
///     counter (TermCounter): Precomputed IC values.
///
/// Returns:
///     float: Similarity score.
///
/// Raises:
///     ValueError: If the method is unknown.
#[pyfunction]
pub fn semantic_similarity(
    id1: &str,
    id2: &str,
    method: &str,
    counter: &TermCounter,
) -> PyResult<f64> {
    let method_enum = SimilarityMethod::from_str(method)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown similarity method: {}", method)))?;

    Ok(method_enum.compute(id1, id2, counter))
}

/// Compute pairwise semantic similarity in batch using a selected method.
///
/// Args:
///     list1 (List[str]): First list of GO term IDs.
///     list2 (List[str]): Second list of GO term IDs.
///     method (str): Name of the similarity method. Options: `"resnik"`, `"lin"`.
///     counter (TermCounter): Precomputed IC values.
///
/// Returns:
///     List[float]: List of similarity scores.
///
/// Raises:
///     ValueError: If input lists differ in length or method is unknown.
#[pyfunction]
pub fn batch_similarity(
    list1: Vec<String>,
    list2: Vec<String>,
    method: &str,
    counter: &TermCounter,
) -> PyResult<Vec<f64>> {
    if list1.len() != list2.len() {
        return Err(PyValueError::new_err("Both lists must be the same length"));
    }

    let method_enum = SimilarityMethod::from_str(method)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown similarity method: {}", method)))?;

    Ok(list1
        .par_iter()
        .zip(list2.par_iter())
        .map(|(id1, id2)| method_enum.compute(id1, id2, counter))
        .collect())
}


/// Compute semantic similarity between genes.
///
/// Args:
///     gene1 (str): Gene symbol of the first gene.
///     gene2 (str): Gene symbol of the second gene.
///     ontology (str): Name of the subontology of GO to use: BP, MF or CC.
///     similarity (str): Name of the similarity method. Options: `"resnik"`, `"lin"`, `"jc"`, `"simrel"`, `"graphic"`, `"iccoef"`.
///     groupwise (str): Combination method to generate the similarities between genes. Options: `"bma"`, `"max"`.
///     counter (TermCounter): Precomputed IC values.
///
/// Returns:
///     List[float]: List of similarity scores.
///
/// Raises:
///     ValueError: If method or combine are unknown.
#[pyfunction]
pub fn compare_genes(
    gene1: &str,
    gene2: &str,
    ontology: String,
    similarity: &str,
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

    let sim_fn = SimilarityMethod::from_str(similarity)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown similarity method: {}", similarity)))?;

    let score = match groupwise.as_str() {
        "max" => {
            f1.iter()
                .flat_map(|id1| {
                    f2.iter()
                        .map(|id2| sim_fn.compute(id1, id2, counter))
                })
                .fold(0.0, f64::max)
        }
        "bma" => {
            let sem1: Vec<f64> = f1.iter()
                .map(|id1| {
                    f2.iter()
                        .map(|id2| sim_fn.compute(id1, id2, counter))
                        .fold(0.0, f64::max)
                })
                .collect();

            let sem2: Vec<f64> = f2.iter()
                .map(|id2| {
                    f1.iter()
                        .map(|id1| sim_fn.compute(id1, id2, counter))
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

/// Compute semantic similarity between genes in batches.
///
/// Args:
///     pairs (List[(str, str)]): List of pairs of genes to calculate the semantic similarity
///     ontology (str): Name of the subontology of GO to use: BP, MF or CC.
///     similarity (str): Name of the similarity method. Options: `"resnik"`, `"lin"`, `"jc"`, `"simrel"`, `"graphic"`, `"iccoef"`.
///     groupwise (str): Combination method to generate the similarities between genes. Options: `"bma"`, `"max"`.
///     counter (TermCounter): Precomputed IC values.
///
/// Returns:
///     List[float]: List of similarity scores.
///
/// Raises:
///     ValueError: If method or combine are unknown.
#[pyfunction]
#[pyo3(signature = (pairs, ontology, similarity, groupwise, counter))]
pub fn compare_gene_pairs_batch(
    pairs: Vec<(String, String)>,
    ontology: String,
    similarity: &str,
    groupwise: String,
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

    let sim_fn = SimilarityMethod::from_str(similarity)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown similarity method: {}", similarity)))?;

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

            match groupwise.as_str() {
                "max" => go1.iter()
                    .flat_map(|id1| go2.iter().map( |id2| sim_fn.compute(id1, id2, counter)))
                    .fold(0.0, f64::max),

                "bma" => {
                    let sem1: Vec<_> = go1.par_iter()
                        .map(|id1| {
                            go2.iter()
                                .map(|id2| sim_fn.compute(id1, id2, counter))
                                .fold(0.0, f64::max)
                        })
                        .collect();

                    let sem2: Vec<_> = go2.par_iter()
                        .map(|id2| {
                            go1.iter()
                                .map(|id1| sim_fn.compute(id1, id2, counter))
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

fn semantic_contributions<'a>(
    go_id: &'a str,
    terms: &'a HashMap<String, crate::go_ontology::GOTerm>,
) -> HashMap<&'a str, f64> {
    let mut contributions = HashMap::new();
    let mut to_visit = vec![(go_id, 1.0)];

    while let Some((current_id, weight)) = to_visit.pop() {
        if weight < 1e-6 || contributions.contains_key(current_id) {
            continue;
        }

        contributions.insert(current_id, weight);

        if let Some(term) = terms.get(current_id) {
            // is_a → 0.8
            for parent in &term.parents {
                to_visit.push((parent, weight * 0.8));
            }
            // part_of → 0.6
            for (rel_type, target) in &term.relationships {
                let rel_weight = match rel_type.as_str() {
                    "part_of" => 0.6,
                    _ => continue,  // skip other relationships for now
                };
                to_visit.push((target, weight * rel_weight));
            }
        }
    }

    contributions
}