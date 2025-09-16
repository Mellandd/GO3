use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use crate::go_loader::TermCounter;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use crate::go_ontology::{deepest_common_ancestor, get_term_by_id, get_terms_or_error, get_gene2go_or_error};
use dashmap::DashMap;

use rayon::ThreadPoolBuilder;

/// Configure the maximum number of threads rayon will use.
///
/// Args:
///     n_threads (int): Number of threads to use. If 0, uses all available cores.
#[pyfunction]
pub fn set_num_threads(n_threads: usize) -> PyResult<()> {
    if n_threads == 0 {
        ThreadPoolBuilder::new()
            .build_global()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    } else {
        ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build_global()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    }
    Ok(())
}

/// Compute the Information Content (IC) of a GO term.
///
/// Arguments
/// ---------
/// go_id : str
///   GO term identifier.
/// counter : TermCounter
///   Precomputed term counter with IC values.
///
/// Returns
/// -------
/// float
///   The IC of the GO term.
#[pyfunction]
#[pyo3(text_signature = "(go_id, counter)")]
pub fn term_ic(go_id: &str, counter: &TermCounter) -> f64 {
    *counter.ic.get(go_id).unwrap_or(&0.0)
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
    TopoICSim,
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
            "topoicsim" => Some(SimilarityMethod::TopoICSim),
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
                if distance.is_infinite() {
                    0.0
                } else {
                    1.0 / (1.0 + distance)
                }
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
                let terms = match crate::go_loader::GO_TERMS_CACHE.get() {
                    Some(lock) => lock.read(),
                    None => return 0.0,
                };
            
                let terms = &*terms;
            
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
            SimilarityMethod::TopoICSim => {
                // Get terms and check namespace
                let terms = match crate::go_loader::GO_TERMS_CACHE.get() {
                    Some(lock) => lock.read(),
                    None => return 0.0,
                };
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
                // Disjunctive common ancestors
                let dca_set = disjunctive_common_ancestors(id1, id2, &terms);
                if dca_set.is_empty() {
                    return 0.0;
                }
                // Find all roots
                let roots = find_roots(&terms);
                if roots.is_empty() {
                    return 0.0;
                }
                let mut min_d = f64::INFINITY;
                for x in dca_set {
                    // Weighted shortest path from t1 to x and t2 to x
                    let wsp1 = weighted_shortest_path_iic(id1, &x, &terms, counter);
                    let wsp2 = weighted_shortest_path_iic(id2, &x, &terms, counter);
                    if wsp1.is_none() || wsp2.is_none() {
                        continue;
                    }
                    let wsp = wsp1.unwrap() + wsp2.unwrap();
                    // Weighted longest path from x to any root (take the max over all roots)
                    let mut max_wlp = None;
                    for root in &roots {
                        if let Some(wlp) = weighted_longest_path_iic(&x, root, &terms, counter) {
                            max_wlp = Some(max_wlp.map_or(wlp, |m: f64| m.max(wlp)));
                        }
                    }
                    let wlp = match max_wlp {
                        Some(val) if val > 0.0 => val,
                        _ => continue,
                    };
                    let d = wsp / wlp;
                    if d < min_d {
                        min_d = d;
                    }
                }
                if !min_d.is_finite() {
                    return 0.0;
                }
                // Similarity formula: 1 - (arctan(D) / (pi/2))
                let sim = 1.0 - (min_d.atan() / (std::f64::consts::FRAC_PI_2));
                if sim.is_finite() && sim > 0.0 { sim } else { 0.0 }
            }
        }
    }
}

/// Compute semantic similarity between two GO terms using a selected method.
///
/// Arguments
/// ---------
/// id1 : str
///   First GO term ID.
/// id2 : str
///   Second GO term ID.
/// method : str
///   Name of the similarity method. Options: "resnik", "lin", etc.
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
///   If the method is unknown.
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
/// Arguments
/// ---------
/// list1 : list of str
///   First list of GO term IDs.
/// list2 : list of str
///   Second list of GO term IDs.
/// method : str
///   Name of the similarity method.
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
///   If input lists differ in length or method is unknown.
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

    // 1. Collect all unique pairs (order them to avoid (a,b) vs (b,a) duplicates)
    let unique_pairs: HashSet<(String, String)> = list1.iter().zip(list2.iter())
        .map(|(a, b)| {
            if a <= b {
                (a.clone(), b.clone())
            } else {
                (b.clone(), a.clone())
            }
        })
        .collect();

    // 2. Compute similarity for each unique pair in parallel
    let sim_map: HashMap<(String, String), f64> = unique_pairs
        .par_iter()
        .map(|(a, b)| {
            let sim = method_enum.compute(a, b, counter);
            ((a.clone(), b.clone()), sim)
        })
        .collect();

    // 3. For each original pair, look up the result (parallelized)
    let result: Vec<f64> = list1.par_iter().zip(list2.par_iter())
        .map(|(a, b)| {
            let key = if a <= b { (a.clone(), b.clone()) } else { (b.clone(), a.clone()) };
            *sim_map.get(&key).unwrap_or(&0.0)
        })
        .collect();

    Ok(result)
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
///   Combination method to generate the similarities between genes. Options: "bma", "max".
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
            f1.par_iter()
                .map(|id1| {
                    f2.par_iter()
                        .map(|id2| sim_fn.compute(id1, id2, counter))
                        .reduce(|| 0.0, f64::max)
                })
                .reduce(|| 0.0, f64::max)
        }
        "bma" => {
            let sem1: Vec<f64> = f1.par_iter()
                .map(|id1| {
                    f2.par_iter()
                        .map(|id2| sim_fn.compute(id1, id2, counter))
                        .reduce(|| 0.0, f64::max)
                })
                .collect();

            let sem2: Vec<f64> = f2.par_iter()
                .map(|id2| {
                    f1.par_iter()
                        .map(|id1| sim_fn.compute(id1, id2, counter))
                        .reduce(|| 0.0, f64::max)
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
/// Arguments
/// ---------
/// pairs : list of (str, str)
///   List of pairs of genes to calculate the semantic similarity
/// ontology : str
///   Name of the subontology of GO to use: BP, MF or CC.
/// similarity : str
///   Name of the similarity method.
/// groupwise : str
///   Combination method to generate the similarities between genes. Options: "bma", "max".
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
                "max" => go1.par_iter()
                    .map(|id1| go2.par_iter().map(|id2| sim_fn.compute(id1, id2, counter)).reduce(|| 0.0, f64::max))
                    .reduce(|| 0.0, f64::max),
                "bma" => {
                    let sem1: Vec<_> = go1.par_iter()
                        .map(|id1| go2.par_iter().map(|id2| sim_fn.compute(id1, id2, counter)).reduce(|| 0.0, f64::max))
                        .collect();
                    let sem2: Vec<_> = go2.par_iter()
                        .map(|id2| go1.par_iter().map(|id1| sim_fn.compute(id1, id2, counter)).reduce(|| 0.0, f64::max))
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

fn semantic_contributions(
    go_id: &str,
    terms: &HashMap<String, crate::go_ontology::GOTerm>,
) -> HashMap<String, f64> {
    // Static cache for memoization
    static SEMANTIC_CONTRIB_CACHE: once_cell::sync::Lazy<DashMap<String, HashMap<String, f64>>> = once_cell::sync::Lazy::new(|| DashMap::new());

    // Check cache first
    if let Some(cached) = SEMANTIC_CONTRIB_CACHE.get(go_id) {
        return cached.clone();
    }

    let mut contributions = HashMap::default();
    let mut to_visit = vec![(go_id, 1.0)];

    while let Some((current_id, weight)) = to_visit.pop() {
        if weight < 1e-6 || contributions.contains_key(current_id) {
            continue;
        }

        contributions.insert(current_id.to_string(), weight);

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

    // Store in cache as HashMap<String, f64>
    SEMANTIC_CONTRIB_CACHE.insert(go_id.to_string(), contributions.clone());
    contributions
}

// --- TopoICSim and helper functions ---

/// Compute the Inverse Information Content (IIC) for a term.
///
/// Arguments
/// ---------
/// go_id : str
///   GO term ID.
/// counter : TermCounter
///   Precomputed term counter with IC values.
///
/// Returns
/// -------
/// float
///   The IIC value for the term.
fn iic(go_id: &str, counter: &TermCounter) -> f64 {
    let ic = *counter.ic.get(go_id).unwrap_or(&0.0);
    if ic > 0.0 {
        1.0 / ic
    } else {
        // If IC is zero, treat as very large (effectively infinite path weight)
        1e12
    }
}

/// Compute the weighted shortest path (sum of IICs) from source to target (ancestor) in the GO DAG.
///
/// Arguments
/// ---------
/// source : str
///   Source GO term ID.
/// target : str
///   Target GO term ID (ancestor).
/// terms : dict
///   Map of GO terms.
/// counter : TermCounter
///   Precomputed term counter with IC values.
///
/// Returns
/// -------
/// Option<float>
///   Minimum sum of IICs along any path from source to target, or None if not connected.
fn weighted_shortest_path_iic(
    source: &str,
    target: &str,
    terms: &HashMap<String, crate::go_ontology::GOTerm>,
    counter: &TermCounter,
) -> Option<f64> {
    use std::collections::{BinaryHeap, HashMap};
    use std::cmp::Ordering;
    #[derive(Copy, Clone, PartialEq)]
    struct State {
        cost: f64,
        node: usize,
    }
    impl Eq for State {}
    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            // Reverse order for min-heap
            other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
        }
    }
    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    // Map node ids to indices for fast lookup
    let mut id2idx = HashMap::new();
    let mut idx2id = Vec::new();
    for (i, id) in terms.keys().enumerate() {
        id2idx.insert(id.as_str(), i);
        idx2id.push(id.as_str());
    }
    let src = match id2idx.get(source) {
        Some(&i) => i,
        None => return None,
    };
    let tgt = match id2idx.get(target) {
        Some(&i) => i,
        None => return None,
    };
    let mut dist = vec![f64::INFINITY; idx2id.len()];
    dist[src] = iic(source, counter);
    let mut heap = BinaryHeap::new();
    heap.push(State { cost: dist[src], node: src });
    while let Some(State { cost, node }) = heap.pop() {
        let node_id = idx2id[node];
        if node == tgt {
            return Some(cost);
        }
        if cost > dist[node] {
            continue;
        }
        if let Some(term) = terms.get(node_id) {
            for parent in &term.parents {
                if let Some(&parent_idx) = id2idx.get(parent.as_str()) {
                    let next = cost + iic(parent, counter);
                    if next < dist[parent_idx] {
                        dist[parent_idx] = next;
                        heap.push(State { cost: next, node: parent_idx });
                    }
                }
            }
        }
    }
    None
}

/// Compute the weighted longest path (sum of IICs) from source to target (ancestor) in the GO DAG.
///
/// Arguments
/// ---------
/// source : str
///   Source GO term ID.
/// target : str
///   Target GO term ID (ancestor).
/// terms : dict
///   Map of GO terms.
/// counter : TermCounter
///   Precomputed term counter with IC values.
///
/// Returns
/// -------
/// Option<float>
///   Maximum sum of IICs along any path from source to target, or None if not connected.
fn weighted_longest_path_iic(
    source: &str,
    target: &str,
    terms: &HashMap<String, crate::go_ontology::GOTerm>,
    counter: &TermCounter,
) -> Option<f64> {
    // Memoization
    use std::collections::HashMap as StdHashMap;
    fn dfs(
        node: &str,
        target: &str,
        terms: &HashMap<String, crate::go_ontology::GOTerm>,
        counter: &TermCounter,
        memo: &mut StdHashMap<String, Option<f64>>,
    ) -> Option<f64> {
        if node == target {
            return Some(iic(node, counter));
        }
        if let Some(&val) = memo.get(node) {
            return val;
        }
        let mut max_path = None;
        if let Some(term) = terms.get(node) {
            for parent in &term.parents {
                if let Some(sub) = dfs(parent, target, terms, counter, memo) {
                    let total = iic(node, counter) + sub;
                    max_path = Some(max_path.map_or(total, |m: f64| m.max(total)));
                }
            }
        }
        memo.insert(node.to_string(), max_path);
        max_path
    }
    let mut memo = StdHashMap::new();
    dfs(source, target, terms, counter, &mut memo)
}

/// Compute the Disjunctive Common Ancestor set for two terms (as per TopoICSim paper).
///
/// Arguments
/// ---------
/// id1 : str
///   First GO term ID.
/// id2 : str
///   Second GO term ID.
/// terms : dict
///   Map of GO terms.
///
/// Returns
/// -------
/// list of str
///   List of disjunctive common ancestor GO term IDs.
fn disjunctive_common_ancestors(
    id1: &str,
    id2: &str,
    terms: &HashMap<String, crate::go_ontology::GOTerm>,
) -> Vec<String> {
    use std::collections::HashSet;
    // 1. Get all common ancestors
    let ancestors1 = crate::go_ontology::collect_ancestors(id1, terms);
    let ancestors2 = crate::go_ontology::collect_ancestors(id2, terms);
    let common: HashSet<_> = ancestors1.intersection(&ancestors2).cloned().collect();
    // 2. For each x in common, check if no child of x is also in common
    let mut dca = Vec::new();
    for x in &common {
        let is_disjunctive = terms.get(x).map_or(false, |term| {
            term.children.iter().all(|c| !common.contains(c))
        });
        if is_disjunctive {
            dca.push(x.clone());
        }
    }
    dca
}

/// Find all root terms (terms with no parents) in the ontology.
///
/// Arguments
/// ---------
/// terms : dict
///   Map of GO terms.
///
/// Returns
/// -------
/// list of str
///   List of root GO term IDs.
fn find_roots(terms: &HashMap<String, crate::go_ontology::GOTerm>) -> Vec<String> {
    terms.iter()
        .filter(|(_, term)| term.parents.is_empty())
        .map(|(id, _)| id.clone())
        .collect()
}