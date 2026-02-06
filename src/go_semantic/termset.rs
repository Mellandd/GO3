use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::go_loader::TermCounter;
use crate::go_ontology::{collect_ancestors, get_terms_or_error, GOTerm};

use super::similarity::{term_ic, SimilarityMethod};

pub(crate) fn termset_similarity_internal_with_method(
    terms1: &[String],
    terms2: &[String],
    sim_fn: Option<SimilarityMethod>,
    groupwise: &str,
    counter: &TermCounter,
    ontology_terms: &HashMap<String, GOTerm>,
) -> PyResult<f64> {
    if terms1.is_empty() || terms2.is_empty() {
        return Ok(0.0);
    }

    if groupwise == "simgic" {
        // Collect all ancestors for each set
        let mut ancestors1: HashSet<String> = HashSet::default();
        for t in terms1 {
            let ancs = collect_ancestors(t, ontology_terms);
            for a in ancs {
                ancestors1.insert(a);
            }
        }
        let mut ancestors2: HashSet<String> = HashSet::default();
        for t in terms2 {
            let ancs = collect_ancestors(t, ontology_terms);
            for a in ancs {
                ancestors2.insert(a);
            }
        }
        
        // Compute Jaccard Index weighted by IC
        let mut intersection_ic = 0.0;
        let mut union_ic = 0.0;
        
        let all_ancestors: HashSet<&String> = ancestors1.union(&ancestors2).collect();

        for term in all_ancestors {
            let ic = term_ic(term, counter);
            let in_1 = ancestors1.contains(term);
            let in_2 = ancestors2.contains(term);
            
            if in_1 && in_2 {
                intersection_ic += ic;
            }
            if in_1 || in_2 {
                union_ic += ic;
            }
        }
        
        if union_ic == 0.0 {
            return Ok(0.0);
        }
        return Ok(intersection_ic / union_ic);
    }
    
    // For other methods, we need the pairwise similarity function
    let sim_fn = sim_fn.ok_or_else(|| {
        PyValueError::new_err("similarity argument is required for this groupwise method")
    })?;

    // Avoid nested rayon parallelism when this function is called from an already-parallel
    // context (e.g. gene_distance_matrix or compare_gene_pairs_batch). In those cases, the
    // outer loop should control parallelism, and we compute the termset score serially.
    let use_parallel = rayon::current_thread_index().is_none();

    match groupwise {
        "max" => {
            let max_val = if use_parallel {
                terms1
                    .par_iter()
                    .map(|id1| {
                        terms2
                            .iter()
                            .map(|id2| sim_fn.compute(id1, id2, counter))
                            .fold(0.0, f64::max)
                    })
                    .reduce(|| 0.0, f64::max)
            } else {
                let mut max_val: f64 = 0.0;
                for id1 in terms1 {
                    for id2 in terms2 {
                        max_val = max_val.max(sim_fn.compute(id1, id2, counter));
                    }
                }
                max_val
            };
            Ok(max_val)
        }
        "bma" => {
            let total = (terms1.len() + terms2.len()) as f64;
            if total == 0.0 {
                return Ok(0.0);
            }

            // Compute row maxima for terms1 and column maxima for terms2 in a single pass
            // over the cartesian product. This avoids doing 2× work for symmetric similarities.
            if use_parallel {
                let (sum_row_max, col_max) = terms1
                    .par_iter()
                    .fold(
                        || (0.0_f64, vec![0.0_f64; terms2.len()]),
                        |(sum, mut col_max), id1| {
                            let mut row_max: f64 = 0.0;
                            for (j, id2) in terms2.iter().enumerate() {
                                let s = sim_fn.compute(id1, id2, counter);
                                if s > row_max {
                                    row_max = s;
                                }
                                if s > col_max[j] {
                                    col_max[j] = s;
                                }
                            }
                            (sum + row_max, col_max)
                        },
                    )
                    .reduce(
                        || (0.0_f64, vec![0.0_f64; terms2.len()]),
                        |(sum_a, mut col_a), (sum_b, col_b)| {
                            for (a, b) in col_a.iter_mut().zip(col_b.iter()) {
                                if *b > *a {
                                    *a = *b;
                                }
                            }
                            (sum_a + sum_b, col_a)
                        },
                    );
                let sum_col_max: f64 = col_max.into_iter().sum();
                Ok((sum_row_max + sum_col_max) / total)
            } else {
                let mut col_max = vec![0.0_f64; terms2.len()];
                let mut sum_row_max: f64 = 0.0;

                for id1 in terms1 {
                    let mut row_max: f64 = 0.0;
                    for (j, id2) in terms2.iter().enumerate() {
                        let s = sim_fn.compute(id1, id2, counter);
                        if s > row_max {
                            row_max = s;
                        }
                        if s > col_max[j] {
                            col_max[j] = s;
                        }
                    }
                    sum_row_max += row_max;
                }

                let sum_col_max: f64 = col_max.into_iter().sum();
                Ok((sum_row_max + sum_col_max) / total)
            }
        }
        "avg" => {
             let count = (terms1.len() * terms2.len()) as f64;
             if count == 0.0 {
                 return Ok(0.0);
             }
             // sum( sim(t1, t2) ) / (N*M)
             let total_sim: f64 = if use_parallel {
                 terms1
                     .par_iter()
                     .map(|id1| {
                         terms2
                             .iter()
                             .map(|id2| sim_fn.compute(id1, id2, counter))
                             .sum::<f64>()
                     })
                     .sum()
             } else {
                 let mut total = 0.0;
                 for id1 in terms1 {
                     for id2 in terms2 {
                         total += sim_fn.compute(id1, id2, counter);
                     }
                 }
                 total
             };
                 
             Ok(total_sim / count)
        }
        "hausdorff" => {
            // min( min_a max_b sim(a, b), min_b max_a sim(b, a) )

            let (min_row_max, col_max) = if use_parallel {
                terms1
                    .par_iter()
                    .fold(
                        || (f64::INFINITY, vec![0.0_f64; terms2.len()]),
                        |(min_row, mut col_max), id1| {
                            let mut row_max: f64 = 0.0;
                            for (j, id2) in terms2.iter().enumerate() {
                                let s = sim_fn.compute(id1, id2, counter);
                                if s > row_max {
                                    row_max = s;
                                }
                                if s > col_max[j] {
                                    col_max[j] = s;
                                }
                            }
                            (min_row.min(row_max), col_max)
                        },
                    )
                    .reduce(
                        || (f64::INFINITY, vec![0.0_f64; terms2.len()]),
                        |(min_a, mut col_a), (min_b, col_b)| {
                            for (a, b) in col_a.iter_mut().zip(col_b.iter()) {
                                if *b > *a {
                                    *a = *b;
                                }
                            }
                            (min_a.min(min_b), col_a)
                        },
                    )
            } else {
                let mut col_max = vec![0.0_f64; terms2.len()];
                let mut min_row_max = f64::INFINITY;

                for id1 in terms1 {
                    let mut row_max: f64 = 0.0;
                    for (j, id2) in terms2.iter().enumerate() {
                        let s = sim_fn.compute(id1, id2, counter);
                        if s > row_max {
                            row_max = s;
                        }
                        if s > col_max[j] {
                            col_max[j] = s;
                        }
                    }
                    min_row_max = min_row_max.min(row_max);
                }

                (min_row_max, col_max)
            };

            let min_col_max: f64 = col_max.into_iter().fold(f64::INFINITY, f64::min);

            if min_row_max.is_infinite() || min_col_max.is_infinite() {
                Ok(0.0)
            } else {
                Ok(min_row_max.min(min_col_max))
            }
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!("Unknown groupwise strategy: {}", groupwise))),
    }
}

/// Internal helper to compute similarity between two sets of GO terms.
pub(crate) fn termset_similarity_internal(
    terms1: &[String],
    terms2: &[String],
    similarity: &str,
    groupwise: &str,
    counter: &TermCounter,
    ontology_terms: &HashMap<String, GOTerm>,
) -> PyResult<f64> {
    let sim_fn = if groupwise == "simgic" {
        None
    } else {
        Some(
            SimilarityMethod::from_str(similarity)
                .ok_or_else(|| PyValueError::new_err(format!("Unknown similarity method: {}", similarity)))?
        )
    };
    termset_similarity_internal_with_method(terms1, terms2, sim_fn, groupwise, counter, ontology_terms)
}


/// Compute semantic similarity between two sets of GO terms.
///
/// Arguments
/// ---------
/// terms1 : list of str
///   First list of GO term IDs.
/// terms2 : list of str
///   Second list of GO term IDs.
/// term_similarity : str
///   Name of the pairwise similarity method.
/// groupwise : str
///   Groupwise combination method. Options: "bma", "max", "avg", "hausdorff", "simgic".
/// counter : TermCounter
///   Precomputed IC values.
///
/// Returns
/// -------
/// float
///   Similarity score.
#[pyfunction]
#[pyo3(signature = (terms1, terms2, term_similarity="lin", groupwise="bma", counter=None))]
pub fn termset_similarity(
    terms1: Vec<String>,
    terms2: Vec<String>,
    term_similarity: &str,
    groupwise: &str,
    counter: Option<&TermCounter>,
) -> PyResult<f64> {
     let c = counter.ok_or_else(|| PyValueError::new_err("counter argument is required"))?;
     let terms_lock = get_terms_or_error()?;
     termset_similarity_internal(&terms1, &terms2, term_similarity, groupwise, c, &terms_lock)
}
