use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::go_loader::TermCounter;
use crate::go_ontology::{get_gene2go_or_error, get_terms_or_error};

use super::similarity::SimilarityMethod;
use super::termset::{termset_similarity_internal, termset_similarity_internal_with_method};

#[derive(Debug, Clone, Copy, PartialEq)]
enum DistanceTransform {
    OneMinus,
    Reciprocal,
    MaxMinus,
}

fn is_valid_groupwise(groupwise: &str) -> bool {
    matches!(groupwise, "bma" | "max" | "avg" | "hausdorff" | "simgic")
}

fn is_normalized_similarity(similarity: &str, groupwise: &str) -> bool {
    if groupwise == "simgic" {
        return true;
    }
    matches!(
        similarity.to_ascii_lowercase().as_str(),
        "lin" | "wang" | "simrel" | "topoicsim"
    )
}

fn resolve_distance_transform(
    distance_transform: &str,
    similarity: &str,
    groupwise: &str,
) -> PyResult<DistanceTransform> {
    match distance_transform.to_ascii_lowercase().as_str() {
        "auto" => {
            if is_normalized_similarity(similarity, groupwise) {
                Ok(DistanceTransform::OneMinus)
            } else {
                Ok(DistanceTransform::MaxMinus)
            }
        }
        "one_minus" | "one-minus" | "1-sim" | "1_minus" => Ok(DistanceTransform::OneMinus),
        "reciprocal" | "inv" | "inverse" => Ok(DistanceTransform::Reciprocal),
        "max_minus" | "max-minus" | "max" => Ok(DistanceTransform::MaxMinus),
        _ => Err(PyValueError::new_err(format!(
            "Unknown distance_transform '{}'. Options: auto, one_minus, reciprocal, max_minus",
            distance_transform
        ))),
    }
}

fn ontology_namespace(ontology: &str) -> PyResult<&'static str> {
    match ontology.to_ascii_uppercase().as_str() {
        "BP" => Ok("biological_process"),
        "MF" => Ok("molecular_function"),
        "CC" => Ok("cellular_component"),
        _ => Err(PyValueError::new_err(format!(
            "Invalid ontology '{}'. Must be 'BP', 'MF', or 'CC'",
            ontology
        ))),
    }
}

/// Compute semantic similarity between genes.
///
/// Arguments
/// ---------
/// gene1 : str
///   Gene symbol of the first gene.
/// gene2 : str
///   Gene symbol of the second gene.
/// ontology : str
///   Name of the subontology of GO to use: BP, MF or CC.
/// similarity : str
///   Name of the similarity method.
/// groupwise : str
///   Combination method to generate the similarities between genes. Options: "bma", "max", "avg", "hausdorff", "simgic".
/// counter : TermCounter
///   Precomputed IC values.
///
/// Returns
/// -------
/// float
///   Similarity score.
///
/// Raises
/// ------
/// ValueError
///   If method or combine are unknown.
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
    let ns = ontology_namespace(&ontology)?;
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

    if f1.is_empty() || f2.is_empty() {
        return Ok(0.0);
    }
 
    termset_similarity_internal(&f1, &f2, similarity, &groupwise, counter, &terms)
}

/// Compute semantic similarity between genes in batches.
///
/// Arguments
/// ---------
/// pairs : list of (str, str)
///   List of pairs of genes to calculate the semantic similarity
/// ontology : str
///   Name of the subontology of GO to use: BP, MF or CC.
/// similarity : str
///   Name of the similarity method.
/// groupwise : str
///   Combination method to generate the similarities between genes. Options: "bma", "max", "avg", "hausdorff", "simgic".
/// counter : TermCounter
///   Precomputed IC values.
///
/// Returns
/// -------
/// list of float
///   List of similarity scores.
///
/// Raises
/// ------
/// ValueError
///   If method or combine are unknown.
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
    let ns = ontology_namespace(&ontology)?;
    let sim_fn = if groupwise == "simgic" {
        None
    } else {
        SimilarityMethod::from_str(similarity)
    };

    let mut unique_genes: HashSet<String> = HashSet::default();
    for (g1, g2) in &pairs {
        unique_genes.insert(g1.clone());
        unique_genes.insert(g2.clone());
    }

    let gene_terms: HashMap<String, Vec<String>> = unique_genes
        .into_iter()
        .map(|gene| {
            let filtered: Vec<String> = gene2go
                .get(&gene)
                .into_iter()
                .flatten()
                .filter(|go| {
                    terms
                        .get(go.as_str())
                        .map_or(false, |t| t.namespace.eq_ignore_ascii_case(ns))
                })
                .cloned()
                .collect();
            (gene, filtered)
        })
        .collect();

    let empty: Vec<String> = Vec::new();
    let scores: Vec<f64> = pairs
        .par_iter()
        .map(|(g1, g2)| {
            let go1 = gene_terms.get(g1).unwrap_or(&empty);
            let go2 = gene_terms.get(g2).unwrap_or(&empty);

            if go1.is_empty() || go2.is_empty() {
                return 0.0;
            }

            termset_similarity_internal_with_method(go1, go2, sim_fn, &groupwise, counter, &terms)
                .unwrap_or(0.0)
        })
        .collect();

    Ok(scores)
}

/// Compute a gene-to-gene distance matrix using GO semantic similarity.
///
/// Arguments
/// ---------
/// genes : Optional[list[str]]
///   List of genes to include. If None, uses all genes with annotations.
/// ontology : str
///   Name of the subontology of GO to use: BP, MF or CC.
/// similarity : str
///   Name of the similarity method.
/// groupwise : str
///   Combination method to generate the similarities between genes. Options: "bma", "max", "avg", "hausdorff", "simgic".
/// counter : TermCounter
///   Precomputed IC values.
/// distance_transform : str
///   How to convert similarity to distance. Options: "auto", "one_minus", "reciprocal", "max_minus".
///
/// Returns
/// -------
/// (list[str], list[list[float]])
///   Tuple with the gene order and a square distance matrix.
#[pyfunction]
#[pyo3(signature = (genes=None, ontology="BP", similarity="lin", groupwise="bma", counter=None, distance_transform="auto"))]
pub fn gene_distance_matrix(
    genes: Option<Vec<String>>,
    ontology: &str,
    similarity: &str,
    groupwise: &str,
    counter: Option<&TermCounter>,
    distance_transform: &str,
) -> PyResult<(Vec<String>, Vec<Vec<f64>>)> {
    let counter = counter.ok_or_else(|| PyValueError::new_err("counter argument is required"))?;
    if !is_valid_groupwise(groupwise) {
        return Err(PyValueError::new_err(format!(
            "Unknown groupwise strategy: {}",
            groupwise
        )));
    }

    let terms = get_terms_or_error()?;
    let gene2go = get_gene2go_or_error()?;
    let ns = ontology_namespace(ontology)?;

    let gene_list = match genes {
        Some(list) => list,
        None => {
            let mut all: Vec<String> = gene2go.keys().cloned().collect();
            all.sort();
            all
        }
    };

    if gene_list.is_empty() {
        return Ok((gene_list, Vec::new()));
    }

    let missing: Vec<String> = gene_list
        .iter()
        .filter(|g| !gene2go.contains_key(*g))
        .cloned()
        .collect();
    if !missing.is_empty() {
        return Err(PyValueError::new_err(format!(
            "Genes not found in mapping: {}",
            missing.join(", ")
        )));
    }

    let sim_fn = if groupwise == "simgic" {
        None
    } else {
        Some(
            SimilarityMethod::from_str(similarity)
                .ok_or_else(|| PyValueError::new_err(format!("Unknown similarity method: {}", similarity)))?
        )
    };

    let gene_terms: Vec<Vec<String>> = gene_list
        .par_iter()
        .map(|gene| {
            let terms_for_gene = gene2go.get(gene).unwrap();
            terms_for_gene
                .iter()
                .filter(|go| {
                    terms
                        .get(go.as_str())
                        .map_or(false, |t| t.namespace.eq_ignore_ascii_case(ns))
                })
                .cloned()
                .collect()
        })
        .collect();

    let n = gene_list.len();
    let mut matrix = vec![vec![0.0; n]; n];

    matrix
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, row)| {
            for j in i..n {
                let sim = termset_similarity_internal_with_method(
                    &gene_terms[i],
                    &gene_terms[j],
                    sim_fn,
                    groupwise,
                    counter,
                    &terms,
                )
                .unwrap_or(0.0);
                row[j] = sim;
            }
        });

    for i in 0..n {
        for j in 0..i {
            let v = matrix[j][i];
            matrix[i][j] = v;
        }
    }

    let transform = resolve_distance_transform(distance_transform, similarity, groupwise)?;
    match transform {
        DistanceTransform::MaxMinus => {
            let mut max_sim = 0.0;
            for row in &matrix {
                for &v in row {
                    if v > max_sim {
                        max_sim = v;
                    }
                }
            }
            matrix.par_iter_mut().for_each(|row| {
                for v in row.iter_mut() {
                    let d = max_sim - *v;
                    *v = if d < 0.0 { 0.0 } else { d };
                }
            });
        }
        DistanceTransform::OneMinus => {
            matrix.par_iter_mut().for_each(|row| {
                for v in row.iter_mut() {
                    let d = 1.0 - *v;
                    *v = if d < 0.0 { 0.0 } else { d };
                }
            });
        }
        DistanceTransform::Reciprocal => {
            matrix.par_iter_mut().for_each(|row| {
                for v in row.iter_mut() {
                    *v = 1.0 / (1.0 + *v);
                }
            });
        }
    }

    for i in 0..n {
        matrix[i][i] = 0.0;
    }

    Ok((gene_list, matrix))
}
