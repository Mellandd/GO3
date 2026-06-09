use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::go_loader::TermCounter;
use crate::go_ontology::{get_gene2go_or_error, get_terms_or_error, GOTerm};

use super::similarity::{term_ic, SimilarityMethod};
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

fn apply_distance_transform(matrix: &mut [Vec<f64>], transform: DistanceTransform) {
    match transform {
        DistanceTransform::MaxMinus => {
            let mut max_sim = 0.0;
            for row in matrix.iter() {
                for &value in row {
                    if value > max_sim {
                        max_sim = value;
                    }
                }
            }
            matrix.par_iter_mut().for_each(|row| {
                for value in row.iter_mut() {
                    let distance = max_sim - *value;
                    *value = if distance < 0.0 { 0.0 } else { distance };
                }
            });
        }
        DistanceTransform::OneMinus => {
            matrix.par_iter_mut().for_each(|row| {
                for value in row.iter_mut() {
                    let distance = 1.0 - *value;
                    *value = if distance < 0.0 { 0.0 } else { distance };
                }
            });
        }
        DistanceTransform::Reciprocal => {
            matrix.par_iter_mut().for_each(|row| {
                for value in row.iter_mut() {
                    *value = 1.0 / (1.0 + *value);
                }
            });
        }
    }
}

fn zero_diagonal(matrix: &mut [Vec<f64>]) {
    for (i, row) in matrix.iter_mut().enumerate() {
        row[i] = 0.0;
    }
}

fn resolve_similarity_method(
    similarity: &str,
    groupwise: &str,
) -> PyResult<Option<SimilarityMethod>> {
    if !is_valid_groupwise(groupwise) {
        return Err(PyValueError::new_err(format!(
            "Unknown groupwise strategy: {}",
            groupwise
        )));
    }

    if groupwise == "simgic" {
        Ok(None)
    } else {
        SimilarityMethod::from_str(similarity)
            .map(Some)
            .ok_or_else(|| {
                PyValueError::new_err(format!("Unknown similarity method: {}", similarity))
            })
    }
}

fn validate_gene_groupwise(groupwise: &str) -> PyResult<()> {
    if !is_valid_groupwise(groupwise) {
        return Err(PyValueError::new_err(format!(
            "Unknown groupwise strategy: {}",
            groupwise
        )));
    }
    if groupwise == "simgic" {
        return Err(PyValueError::new_err(
            "simgic is not valid for direct gene-set aggregation over pairwise gene similarities. Use bma, max, avg, hausdorff, or compare_gene_set_profiles for GO-profile SimGIC.",
        ));
    }
    Ok(())
}

fn filtered_terms_for_gene(
    terms_for_gene: &[String],
    terms: &HashMap<String, GOTerm>,
    ns: &str,
) -> Vec<String> {
    terms_for_gene
        .iter()
        .filter(|go| {
            terms
                .get(go.as_str())
                .map_or(false, |t| t.namespace.eq_ignore_ascii_case(ns))
        })
        .cloned()
        .collect()
}

fn missing_genes(genes: &[String], gene2go: &HashMap<String, Vec<String>>) -> Vec<String> {
    let mut missing: Vec<String> = genes
        .iter()
        .filter(|gene| !gene2go.contains_key(gene.as_str()))
        .cloned()
        .collect();
    missing.sort();
    missing.dedup();
    missing
}

fn ensure_genes_exist(genes: &[String], gene2go: &HashMap<String, Vec<String>>) -> PyResult<()> {
    let missing = missing_genes(genes, gene2go);
    if !missing.is_empty() {
        return Err(PyValueError::new_err(format!(
            "Genes not found in mapping: {}",
            missing.join(", ")
        )));
    }
    Ok(())
}

fn ensure_gene_sets_exist(
    gene_sets: &[(String, Vec<String>)],
    gene2go: &HashMap<String, Vec<String>>,
) -> PyResult<()> {
    let mut missing_set: HashSet<String> = HashSet::default();
    for (_, genes) in gene_sets {
        missing_set.extend(missing_genes(genes, gene2go));
    }
    if !missing_set.is_empty() {
        let mut missing: Vec<String> = missing_set.into_iter().collect();
        missing.sort();
        return Err(PyValueError::new_err(format!(
            "Genes not found in mapping: {}",
            missing.join(", ")
        )));
    }
    Ok(())
}

fn collect_gene_terms_for_genes(
    genes: &[String],
    gene2go: &HashMap<String, Vec<String>>,
    terms: &HashMap<String, GOTerm>,
    ns: &str,
) -> Vec<Vec<String>> {
    genes
        .iter()
        .map(|gene| {
            gene2go
                .get(gene.as_str())
                .map(|terms_for_gene| filtered_terms_for_gene(terms_for_gene, terms, ns))
                .unwrap_or_default()
        })
        .collect()
}

#[derive(Clone)]
struct WeightedTerm {
    id: String,
    weight: f64,
}

fn collect_weighted_gene_set_terms(
    genes: &[String],
    gene2go: &HashMap<String, Vec<String>>,
    terms: &HashMap<String, GOTerm>,
    ns: &str,
) -> Vec<WeightedTerm> {
    let mut weights: HashMap<String, f64> = HashMap::default();
    for gene in genes {
        let Some(terms_for_gene) = gene2go.get(gene.as_str()) else {
            continue;
        };
        let mut seen_for_gene: HashSet<String> = HashSet::default();
        for go in terms_for_gene {
            if terms
                .get(go.as_str())
                .map_or(false, |t| t.namespace.eq_ignore_ascii_case(ns))
                && seen_for_gene.insert(go.clone())
            {
                *weights.entry(go.clone()).or_insert(0.0) += 1.0;
            }
        }
    }

    let mut profile: Vec<WeightedTerm> = weights
        .into_iter()
        .map(|(id, weight)| WeightedTerm { id, weight })
        .collect();
    profile.sort_by(|a, b| a.id.cmp(&b.id));
    profile
}

fn weighted_ancestor_profile(
    profile: &[WeightedTerm],
    ontology_terms: &HashMap<String, GOTerm>,
) -> HashMap<String, f64> {
    let mut weights: HashMap<String, f64> = HashMap::default();
    for term in profile {
        for ancestor in crate::go_ontology::collect_ancestors(&term.id, ontology_terms) {
            *weights.entry(ancestor).or_insert(0.0) += term.weight;
        }
    }
    weights
}

fn weighted_termset_similarity_internal(
    profile1: &[WeightedTerm],
    profile2: &[WeightedTerm],
    sim_fn: Option<SimilarityMethod>,
    groupwise: &str,
    counter: &TermCounter,
    ontology_terms: &HashMap<String, GOTerm>,
) -> PyResult<f64> {
    if profile1.is_empty() || profile2.is_empty() {
        return Ok(0.0);
    }

    if groupwise == "simgic" {
        let ancestors1 = weighted_ancestor_profile(profile1, ontology_terms);
        let ancestors2 = weighted_ancestor_profile(profile2, ontology_terms);
        let all_ancestors: HashSet<&String> = ancestors1.keys().chain(ancestors2.keys()).collect();
        let mut intersection_ic = 0.0;
        let mut union_ic = 0.0;

        for term in all_ancestors {
            let weight1 = ancestors1.get(term).copied().unwrap_or(0.0);
            let weight2 = ancestors2.get(term).copied().unwrap_or(0.0);
            let ic = term_ic(term, counter);
            intersection_ic += weight1.min(weight2) * ic;
            union_ic += weight1.max(weight2) * ic;
        }

        if union_ic == 0.0 {
            return Ok(0.0);
        }
        return Ok(intersection_ic / union_ic);
    }

    let sim_fn = sim_fn.ok_or_else(|| {
        PyValueError::new_err("similarity argument is required for this groupwise method")
    })?;

    match groupwise {
        "max" => {
            let mut max_val: f64 = 0.0;
            for t1 in profile1 {
                for t2 in profile2 {
                    max_val = max_val.max(sim_fn.compute(&t1.id, &t2.id, counter));
                }
            }
            Ok(max_val)
        }
        "avg" => {
            let total_weight1: f64 = profile1.iter().map(|term| term.weight).sum();
            let total_weight2: f64 = profile2.iter().map(|term| term.weight).sum();
            let denominator = total_weight1 * total_weight2;
            if denominator == 0.0 {
                return Ok(0.0);
            }

            let mut total = 0.0;
            for t1 in profile1 {
                for t2 in profile2 {
                    total += t1.weight * t2.weight * sim_fn.compute(&t1.id, &t2.id, counter);
                }
            }
            Ok(total / denominator)
        }
        "bma" => {
            let total_weight1: f64 = profile1.iter().map(|term| term.weight).sum();
            let total_weight2: f64 = profile2.iter().map(|term| term.weight).sum();
            let denominator = total_weight1 + total_weight2;
            if denominator == 0.0 {
                return Ok(0.0);
            }

            let mut sum1 = 0.0;
            for t1 in profile1 {
                let row_max = profile2
                    .iter()
                    .map(|t2| sim_fn.compute(&t1.id, &t2.id, counter))
                    .fold(0.0, f64::max);
                sum1 += t1.weight * row_max;
            }

            let mut sum2 = 0.0;
            for t2 in profile2 {
                let col_max = profile1
                    .iter()
                    .map(|t1| sim_fn.compute(&t1.id, &t2.id, counter))
                    .fold(0.0, f64::max);
                sum2 += t2.weight * col_max;
            }

            Ok((sum1 + sum2) / denominator)
        }
        "hausdorff" => {
            let min_row_max = profile1
                .iter()
                .map(|t1| {
                    profile2
                        .iter()
                        .map(|t2| sim_fn.compute(&t1.id, &t2.id, counter))
                        .fold(0.0, f64::max)
                })
                .fold(f64::INFINITY, f64::min);
            let min_col_max = profile2
                .iter()
                .map(|t2| {
                    profile1
                        .iter()
                        .map(|t1| sim_fn.compute(&t1.id, &t2.id, counter))
                        .fold(0.0, f64::max)
                })
                .fold(f64::INFINITY, f64::min);

            if min_row_max.is_infinite() || min_col_max.is_infinite() {
                Ok(0.0)
            } else {
                Ok(min_row_max.min(min_col_max))
            }
        }
        _ => Err(PyValueError::new_err(format!(
            "Unknown groupwise strategy: {}",
            groupwise
        ))),
    }
}

fn gene_set_similarity_from_gene_terms(
    genes1_terms: &[Vec<String>],
    genes2_terms: &[Vec<String>],
    sim_fn: Option<SimilarityMethod>,
    term_groupwise: &str,
    gene_groupwise: &str,
    counter: &TermCounter,
    ontology_terms: &HashMap<String, GOTerm>,
) -> PyResult<f64> {
    validate_gene_groupwise(gene_groupwise)?;
    if genes1_terms.is_empty() || genes2_terms.is_empty() {
        return Ok(0.0);
    }

    let gene_sim = |terms1: &[String], terms2: &[String]| -> f64 {
        if terms1.is_empty() || terms2.is_empty() {
            return 0.0;
        }
        termset_similarity_internal_with_method(
            terms1,
            terms2,
            sim_fn,
            term_groupwise,
            counter,
            ontology_terms,
        )
        .unwrap_or(0.0)
    };

    match gene_groupwise {
        "max" => {
            let mut max_val: f64 = 0.0;
            for terms1 in genes1_terms {
                for terms2 in genes2_terms {
                    max_val = max_val.max(gene_sim(terms1, terms2));
                }
            }
            Ok(max_val)
        }
        "avg" => {
            let count = (genes1_terms.len() * genes2_terms.len()) as f64;
            if count == 0.0 {
                return Ok(0.0);
            }
            let mut total = 0.0;
            for terms1 in genes1_terms {
                for terms2 in genes2_terms {
                    total += gene_sim(terms1, terms2);
                }
            }
            Ok(total / count)
        }
        "bma" => {
            let total = (genes1_terms.len() + genes2_terms.len()) as f64;
            if total == 0.0 {
                return Ok(0.0);
            }

            let mut col_max = vec![0.0_f64; genes2_terms.len()];
            let mut sum_row_max: f64 = 0.0;
            for terms1 in genes1_terms {
                let mut row_max: f64 = 0.0;
                for (j, terms2) in genes2_terms.iter().enumerate() {
                    let score = gene_sim(terms1, terms2);
                    row_max = row_max.max(score);
                    col_max[j] = col_max[j].max(score);
                }
                sum_row_max += row_max;
            }

            let sum_col_max: f64 = col_max.into_iter().sum();
            Ok((sum_row_max + sum_col_max) / total)
        }
        "hausdorff" => {
            let mut col_max = vec![0.0_f64; genes2_terms.len()];
            let mut min_row_max = f64::INFINITY;

            for terms1 in genes1_terms {
                let mut row_max: f64 = 0.0;
                for (j, terms2) in genes2_terms.iter().enumerate() {
                    let score = gene_sim(terms1, terms2);
                    row_max = row_max.max(score);
                    col_max[j] = col_max[j].max(score);
                }
                min_row_max = min_row_max.min(row_max);
            }

            let min_col_max = col_max.into_iter().fold(f64::INFINITY, f64::min);
            if min_row_max.is_infinite() || min_col_max.is_infinite() {
                Ok(0.0)
            } else {
                Ok(min_row_max.min(min_col_max))
            }
        }
        _ => Err(PyValueError::new_err(format!(
            "Unknown groupwise strategy: {}",
            gene_groupwise
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
    let f1 = filtered_terms_for_gene(g1_terms, &terms, ns);
    let f2 = filtered_terms_for_gene(g2_terms, &terms, ns);

    if f1.is_empty() || f2.is_empty() {
        return Ok(0.0);
    }

    termset_similarity_internal(&f1, &f2, similarity, &groupwise, counter, &terms)
}

/// Compute semantic similarity between two sets of genes.
///
/// Each pair of genes is compared using GO-term semantic similarity, then the
/// resulting gene-by-gene similarity matrix is aggregated with a groupwise
/// strategy across the two gene sets.
///
/// Arguments
/// ---------
/// genes1 : list of str
///   First list of gene symbols.
/// genes2 : list of str
///   Second list of gene symbols.
/// ontology : str
///   Name of the subontology of GO to use: BP, MF or CC.
/// similarity : str
///   Name of the pairwise GO-term similarity method.
/// groupwise : str
///   Groupwise combination method over pairwise gene similarities. Options: "bma", "max", "avg", "hausdorff".
/// counter : TermCounter
///   Precomputed IC values.
/// term_groupwise : Optional[str]
///   Groupwise method used internally when comparing each pair of genes. Defaults to `groupwise`.
///
/// Returns
/// -------
/// float
///   Similarity score.
#[pyfunction]
#[pyo3(signature = (genes1, genes2, ontology="BP", similarity="lin", groupwise="bma", counter=None, term_groupwise=None))]
pub fn compare_gene_sets(
    genes1: Vec<String>,
    genes2: Vec<String>,
    ontology: &str,
    similarity: &str,
    groupwise: &str,
    counter: Option<&TermCounter>,
    term_groupwise: Option<&str>,
) -> PyResult<f64> {
    let counter = counter.ok_or_else(|| PyValueError::new_err("counter argument is required"))?;
    let terms = get_terms_or_error()?;
    let gene2go = get_gene2go_or_error()?;
    let ns = ontology_namespace(ontology)?;
    let term_groupwise = term_groupwise.unwrap_or(groupwise);
    validate_gene_groupwise(groupwise)?;
    let sim_fn = resolve_similarity_method(similarity, term_groupwise)?;

    ensure_genes_exist(&genes1, &gene2go)?;
    ensure_genes_exist(&genes2, &gene2go)?;
    let genes1_terms = collect_gene_terms_for_genes(&genes1, &gene2go, &terms, ns);
    let genes2_terms = collect_gene_terms_for_genes(&genes2, &gene2go, &terms, ns);

    gene_set_similarity_from_gene_terms(
        &genes1_terms,
        &genes2_terms,
        sim_fn,
        term_groupwise,
        groupwise,
        counter,
        &terms,
    )
}

/// Compute semantic similarity between two gene sets via aggregated GO-term profiles.
///
/// Each gene set is converted to a weighted GO-term profile. Term weights are
/// the number of genes in the set annotated to each ontology-filtered GO term.
/// The two weighted GO profiles are then compared directly with the requested
/// groupwise strategy.
#[pyfunction]
#[pyo3(signature = (genes1, genes2, ontology="BP", similarity="lin", groupwise="bma", counter=None))]
pub fn compare_gene_set_profiles(
    genes1: Vec<String>,
    genes2: Vec<String>,
    ontology: &str,
    similarity: &str,
    groupwise: &str,
    counter: Option<&TermCounter>,
) -> PyResult<f64> {
    let counter = counter.ok_or_else(|| PyValueError::new_err("counter argument is required"))?;
    let terms = get_terms_or_error()?;
    let gene2go = get_gene2go_or_error()?;
    let ns = ontology_namespace(ontology)?;
    let sim_fn = resolve_similarity_method(similarity, groupwise)?;

    ensure_genes_exist(&genes1, &gene2go)?;
    ensure_genes_exist(&genes2, &gene2go)?;
    let profile1 = collect_weighted_gene_set_terms(&genes1, &gene2go, &terms, ns);
    let profile2 = collect_weighted_gene_set_terms(&genes2, &gene2go, &terms, ns);

    weighted_termset_similarity_internal(&profile1, &profile2, sim_fn, groupwise, counter, &terms)
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

/// Compute semantic similarity for batches of gene-set pairs.
///
/// Arguments
/// ---------
/// pairs : list of (list[str], list[str])
///   List of gene-set pairs to compare.
/// ontology : str
///   Name of the subontology of GO to use: BP, MF or CC.
/// similarity : str
///   Name of the pairwise GO-term similarity method.
/// groupwise : str
///   Groupwise combination method over pairwise gene similarities. Options: "bma", "max", "avg", "hausdorff".
/// counter : TermCounter
///   Precomputed IC values.
/// term_groupwise : Optional[str]
///   Groupwise method used internally when comparing each pair of genes. Defaults to `groupwise`.
///
/// Returns
/// -------
/// list of float
///   List of similarity scores.
#[pyfunction]
#[pyo3(signature = (pairs, ontology="BP", similarity="lin", groupwise="bma", counter=None, term_groupwise=None))]
pub fn compare_gene_set_pairs_batch(
    pairs: Vec<(Vec<String>, Vec<String>)>,
    ontology: &str,
    similarity: &str,
    groupwise: &str,
    counter: Option<&TermCounter>,
    term_groupwise: Option<&str>,
) -> PyResult<Vec<f64>> {
    let counter = counter.ok_or_else(|| PyValueError::new_err("counter argument is required"))?;
    let gene2go = get_gene2go_or_error()?;
    let terms = get_terms_or_error()?;
    let ns = ontology_namespace(ontology)?;
    let term_groupwise = term_groupwise.unwrap_or(groupwise);
    validate_gene_groupwise(groupwise)?;
    let sim_fn = resolve_similarity_method(similarity, term_groupwise)?;

    let mut unique_genes: HashSet<String> = HashSet::default();
    for (genes1, genes2) in &pairs {
        unique_genes.extend(genes1.iter().cloned());
        unique_genes.extend(genes2.iter().cloned());
    }

    let gene_terms: HashMap<String, Vec<String>> = unique_genes
        .into_iter()
        .map(|gene| {
            let filtered = gene2go
                .get(gene.as_str())
                .map(|terms_for_gene| filtered_terms_for_gene(terms_for_gene, &terms, ns))
                .unwrap_or_default();
            (gene, filtered)
        })
        .collect();

    let scores: Vec<f64> = pairs
        .par_iter()
        .map(|(genes1, genes2)| {
            let genes1_terms: Vec<Vec<String>> = genes1
                .iter()
                .map(|gene| gene_terms.get(gene.as_str()).cloned().unwrap_or_default())
                .collect();
            let genes2_terms: Vec<Vec<String>> = genes2
                .iter()
                .map(|gene| gene_terms.get(gene.as_str()).cloned().unwrap_or_default())
                .collect();

            gene_set_similarity_from_gene_terms(
                &genes1_terms,
                &genes2_terms,
                sim_fn,
                term_groupwise,
                groupwise,
                counter,
                &terms,
            )
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
        Some(SimilarityMethod::from_str(similarity).ok_or_else(|| {
            PyValueError::new_err(format!("Unknown similarity method: {}", similarity))
        })?)
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

    // Parallelize over all (i, j) in the upper triangle (including diagonal) to improve
    // load balancing across threads. This is especially important when each pairwise gene
    // comparison is expensive (large term sets / slower similarity methods).
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (i..n).map(move |j| (i, j))).collect();

    let sims: Vec<f64> = pairs
        .par_iter()
        .map(|(i, j)| {
            termset_similarity_internal_with_method(
                &gene_terms[*i],
                &gene_terms[*j],
                sim_fn,
                groupwise,
                counter,
                &terms,
            )
            .unwrap_or(0.0)
        })
        .collect();

    for ((i, j), sim) in pairs.into_iter().zip(sims.into_iter()) {
        matrix[i][j] = sim;
        matrix[j][i] = sim;
    }

    let transform = resolve_distance_transform(distance_transform, similarity, groupwise)?;
    apply_distance_transform(&mut matrix, transform);
    zero_diagonal(&mut matrix);

    Ok((gene_list, matrix))
}

/// Compute a gene-set-to-gene-set distance matrix using GO semantic similarity.
///
/// Arguments
/// ---------
/// gene_sets : list of (str, list[str])
///   Named gene sets to include. Each item is a pair of set name and gene list.
/// ontology : str
///   Name of the subontology of GO to use: BP, MF or CC.
/// similarity : str
///   Name of the pairwise GO-term similarity method.
/// groupwise : str
///   Groupwise combination method over pairwise gene similarities. Options: "bma", "max", "avg", "hausdorff".
/// counter : TermCounter
///   Precomputed IC values.
/// distance_transform : str
///   How to convert similarity to distance. Options: "auto", "one_minus", "reciprocal", "max_minus".
/// term_groupwise : Optional[str]
///   Groupwise method used internally when comparing each pair of genes. Defaults to `groupwise`.
///
/// Returns
/// -------
/// (list[str], list[list[float]])
///   Tuple with the gene-set names and a square distance matrix.
#[pyfunction]
#[pyo3(signature = (gene_sets, ontology="BP", similarity="lin", groupwise="bma", counter=None, distance_transform="auto", term_groupwise=None))]
pub fn gene_set_distance_matrix(
    gene_sets: Vec<(String, Vec<String>)>,
    ontology: &str,
    similarity: &str,
    groupwise: &str,
    counter: Option<&TermCounter>,
    distance_transform: &str,
    term_groupwise: Option<&str>,
) -> PyResult<(Vec<String>, Vec<Vec<f64>>)> {
    let counter = counter.ok_or_else(|| PyValueError::new_err("counter argument is required"))?;
    let terms = get_terms_or_error()?;
    let gene2go = get_gene2go_or_error()?;
    let ns = ontology_namespace(ontology)?;
    let term_groupwise = term_groupwise.unwrap_or(groupwise);
    validate_gene_groupwise(groupwise)?;
    let sim_fn = resolve_similarity_method(similarity, term_groupwise)?;

    let names: Vec<String> = gene_sets.iter().map(|(name, _)| name.clone()).collect();
    if gene_sets.is_empty() {
        return Ok((names, Vec::new()));
    }

    ensure_gene_sets_exist(&gene_sets, &gene2go)?;

    let mut unique_genes: HashSet<String> = HashSet::default();
    for (_, genes) in &gene_sets {
        unique_genes.extend(genes.iter().cloned());
    }

    let gene_terms: HashMap<String, Vec<String>> = unique_genes
        .into_iter()
        .map(|gene| {
            let filtered = gene2go
                .get(gene.as_str())
                .map(|terms_for_gene| filtered_terms_for_gene(terms_for_gene, &terms, ns))
                .unwrap_or_default();
            (gene, filtered)
        })
        .collect();

    let set_gene_terms: Vec<Vec<Vec<String>>> = gene_sets
        .par_iter()
        .map(|(_, genes)| {
            genes
                .iter()
                .map(|gene| gene_terms.get(gene.as_str()).cloned().unwrap_or_default())
                .collect()
        })
        .collect();

    let n = gene_sets.len();
    let mut matrix = vec![vec![0.0; n]; n];
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (i..n).map(move |j| (i, j))).collect();

    let sims: Vec<f64> = pairs
        .par_iter()
        .map(|(i, j)| {
            gene_set_similarity_from_gene_terms(
                &set_gene_terms[*i],
                &set_gene_terms[*j],
                sim_fn,
                term_groupwise,
                groupwise,
                counter,
                &terms,
            )
            .unwrap_or(0.0)
        })
        .collect();

    for ((i, j), sim) in pairs.into_iter().zip(sims.into_iter()) {
        matrix[i][j] = sim;
        matrix[j][i] = sim;
    }

    let transform = resolve_distance_transform(distance_transform, similarity, term_groupwise)?;
    apply_distance_transform(&mut matrix, transform);
    zero_diagonal(&mut matrix);

    Ok((names, matrix))
}
