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

    match groupwise {
        "max" => {
            let max_val = terms1.par_iter()
                .map(|id1| {
                    terms2.par_iter()
                        .map(|id2| sim_fn.compute(id1, id2, counter))
                        .reduce(|| 0.0, f64::max)
                })
                .reduce(|| 0.0, f64::max);
            Ok(max_val)
        }
        "bma" => {
            let sem1: Vec<f64> = terms1.par_iter()
                .map(|id1| {
                    terms2.par_iter()
                        .map(|id2| sim_fn.compute(id1, id2, counter))
                        .reduce(|| 0.0, f64::max)
                })
                .collect();

            let sem2: Vec<f64> = terms2.par_iter()
                .map(|id2| {
                    terms1.par_iter()
                        .map(|id1| sim_fn.compute(id1, id2, counter))
                        .reduce(|| 0.0, f64::max)
                })
                .collect();

            let total = sem1.len() + sem2.len();
            if total == 0 {
                Ok(0.0)
            } else {
                Ok((sem1.iter().sum::<f64>() + sem2.iter().sum::<f64>()) / total as f64)
            }
        }
        "avg" => {
             let count = (terms1.len() * terms2.len()) as f64;
             if count == 0.0 {
                 return Ok(0.0);
             }
             // sum( sim(t1, t2) ) / (N*M)
             let total_sim: f64 = terms1.par_iter()
                 .map(|id1| {
                     terms2.par_iter()
                        .map(|id2| sim_fn.compute(id1, id2, counter))
                        .sum::<f64>()
                 })
                 .sum();
                 
             Ok(total_sim / count)
        }
        "hausdorff" => {
            // min( min_a max_b sim(a, b), min_b max_a sim(b, a) )
            
            let min_max_1: f64 = terms1.par_iter()
                .map(|id1| {
                     terms2.par_iter()
                        .map(|id2| sim_fn.compute(id1, id2, counter))
                        .reduce(|| 0.0, f64::max)
                })
                .reduce(|| f64::INFINITY, f64::min);
                
            let min_max_2: f64 = terms2.par_iter()
                .map(|id2| {
                     terms1.par_iter()
                        .map(|id1| sim_fn.compute(id1, id2, counter))
                        .reduce(|| 0.0, f64::max)
                })
                .reduce(|| f64::INFINITY, f64::min);
                
             if min_max_1.is_infinite() || min_max_2.is_infinite() {
                 Ok(0.0)
             } else {
                 Ok(min_max_1.min(min_max_2))
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
