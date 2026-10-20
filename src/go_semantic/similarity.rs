use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::go_loader::{GO_TERMS_CACHE, TermCounter};
use crate::go_ontology::{collect_ancestors, deepest_common_ancestor, get_term_by_id, GOTerm};
use dashmap::DashMap;
use std::sync::Arc;

static SEMANTIC_CONTRIB_CACHE: once_cell::sync::Lazy<
    DashMap<String, Arc<HashMap<String, f64>>>,
> = once_cell::sync::Lazy::new(DashMap::new);

pub(crate) fn clear_internal_caches() {
    SEMANTIC_CONTRIB_CACHE.clear();
}

/// Configure the maximum number of threads rayon will use.
///
/// Args:
///     n_threads (int): Number of threads to use. If 0, uses all available cores.
#[pyfunction]
pub fn set_num_threads(n_threads: usize) -> PyResult<()> {
    let builder = if n_threads == 0 {
        ThreadPoolBuilder::new()
    } else {
        ThreadPoolBuilder::new().num_threads(n_threads)
    };

    match builder.build_global() {
        Ok(()) => Ok(()),
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("already been initialized") {
                Ok(())
            } else {
                Err(pyo3::exceptions::PyRuntimeError::new_err(msg))
            }
        }
    }
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


#[derive(Debug, Clone, Copy)]
pub(crate) enum SimilarityMethod {
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
    pub(crate) fn from_str(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
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

    pub(crate) fn compute(&self, id1: &str, id2: &str, counter: &TermCounter) -> f64 {
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
                if id1 == id2 {
                    return 1.0
                }
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
                let (t1, t2) = match (get_term_by_id(id1).ok(), get_term_by_id(id2).ok()) {
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
                let (t1, t2) = match (get_term_by_id(id1).ok(), get_term_by_id(id2).ok()) {
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
                let (t1, t2) = match (get_term_by_id(id1).ok(), get_term_by_id(id2).ok()) {
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
                let (t1, t2) = match (get_term_by_id(id1).ok(), get_term_by_id(id2).ok()) {
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
                let terms = match GO_TERMS_CACHE.get() {
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

                let (small, large) = if sv_a.len() <= sv_b.len() {
                    (&*sv_a, &*sv_b)
                } else {
                    (&*sv_b, &*sv_a)
                };

                let mut numerator = 0.0;
                for (key, w1) in small.iter() {
                    if let Some(w2) = large.get(key) {
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
                let terms = match GO_TERMS_CACHE.get() {
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

fn semantic_contributions(
    go_id: &str,
    terms: &HashMap<String, GOTerm>,
) -> Arc<HashMap<String, f64>> {
    if let Some(cached) = SEMANTIC_CONTRIB_CACHE.get(go_id) {
        return Arc::clone(cached.value());
    }

    let mut contributions = HashMap::default();
    let mut to_visit: Vec<(&str, f64)> = vec![(go_id, 1.0)];

    while let Some((current_id, weight)) = to_visit.pop() {
        if weight < 1e-6 || contributions.contains_key(current_id) {
            continue;
        }

        contributions.insert(current_id.to_string(), weight);

        if let Some(term) = terms.get(current_id) {
            // is_a → 0.8
            for parent in &term.parents {
                to_visit.push((parent.as_str(), weight * 0.8));
            }
            // part_of → 0.6
            for (rel_type, target) in &term.relationships {
                if rel_type == "part_of" {
                    to_visit.push((target.as_str(), weight * 0.6));
                }
            }
        }
    }

    let contributions = Arc::new(contributions);
    SEMANTIC_CONTRIB_CACHE.insert(go_id.to_string(), Arc::clone(&contributions));
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
    terms: &HashMap<String, GOTerm>,
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
    terms: &HashMap<String, GOTerm>,
    counter: &TermCounter,
) -> Option<f64> {
    // Memoization
    use std::collections::HashMap as StdHashMap;
    fn dfs(
        node: &str,
        target: &str,
        terms: &HashMap<String, GOTerm>,
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
    terms: &HashMap<String, GOTerm>,
) -> Vec<String> {
    use std::collections::HashSet;
    // 1. Get all common ancestors
    let ancestors1 = collect_ancestors(id1, terms);
    let ancestors2 = collect_ancestors(id2, terms);
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
fn find_roots(terms: &HashMap<String, GOTerm>) -> Vec<String> {
    terms.iter()
        .filter(|(_, term)| term.parents.is_empty())
        .map(|(id, _)| id.clone())
        .collect()
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

    // Intern terms to indices to avoid cloning Strings into HashSet/HashMap keys.
    let mut term_to_idx: HashMap<&str, usize> = HashMap::default();
    let mut idx_to_term: Vec<&str> = Vec::new();
    for s in list1.iter().chain(list2.iter()) {
        let key = s.as_str();
        if term_to_idx.get(key).is_none() {
            let idx = idx_to_term.len();
            term_to_idx.insert(key, idx);
            idx_to_term.push(key);
        }
    }

    // 1) Collect unique unordered pairs, and keep the per-item lookup key to preserve order.
    let mut unique_pairs: HashSet<(usize, usize)> = HashSet::default();
    let mut pair_indices: Vec<(usize, usize)> = Vec::with_capacity(list1.len());
    for (a, b) in list1.iter().zip(list2.iter()) {
        let i = *term_to_idx
            .get(a.as_str())
            .expect("intern table should contain all terms in list1");
        let j = *term_to_idx
            .get(b.as_str())
            .expect("intern table should contain all terms in list2");
        let key = if i <= j { (i, j) } else { (j, i) };
        unique_pairs.insert(key);
        pair_indices.push(key);
    }

    // 2) Compute similarity for each unique pair in parallel.
    let sim_map: HashMap<(usize, usize), f64> = unique_pairs
        .par_iter()
        .map(|(i, j)| {
            let a = idx_to_term[*i];
            let b = idx_to_term[*j];
            let sim = method_enum.compute(a, b, counter);
            ((*i, *j), sim)
        })
        .collect();

    // 3) For each original pair, look up the result (parallelized).
    let result: Vec<f64> = pair_indices
        .par_iter()
        .map(|(i, j)| *sim_map.get(&(*i, *j)).unwrap_or(&0.0))
        .collect();

    Ok(result)
}

/// Compute an all-vs-all symmetric semantic similarity matrix for a list of GO terms.
///
/// Arguments
/// ---------
/// terms : list of str
///   List of GO term IDs.
/// method : str
///   Name of the similarity method (default: "lin").
/// counter : TermCounter
///   Precomputed IC values.
///
/// Returns
/// -------
/// (list[str], list[list[float]])
///   Tuple with the term order and a square similarity matrix.
#[pyfunction]
#[pyo3(signature = (terms, method="lin", counter=None))]
pub fn term_similarity_matrix(
    terms: Vec<String>,
    method: &str,
    counter: Option<&TermCounter>,
) -> PyResult<(Vec<String>, Vec<Vec<f64>>)> {
    let counter = counter.ok_or_else(|| PyValueError::new_err("counter argument is required"))?;
    let method_enum = SimilarityMethod::from_str(method)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown similarity method: {}", method)))?;

    let n = terms.len();
    if n == 0 {
        return Ok((terms, Vec::new()));
    }

    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (i..n).map(move |j| (i, j))).collect();
    let sims: Vec<f64> = pairs
        .par_iter()
        .map(|(i, j)| method_enum.compute(&terms[*i], &terms[*j], counter))
        .collect();

    let mut matrix = vec![vec![0.0; n]; n];
    for ((i, j), sim) in pairs.into_iter().zip(sims.into_iter()) {
        matrix[i][j] = sim;
        matrix[j][i] = sim;
    }

    Ok((terms, matrix))
}

/// Compute an all-vs-all symmetric distance matrix for a list of GO terms.
///
/// Arguments
/// ---------
/// terms : list of str
///   List of GO term IDs.
/// method : str
///   Name of the similarity method (default: "lin").
/// counter : TermCounter
///   Precomputed IC values.
/// distance_transform : str
///   How to convert similarity to distance. Options: "auto", "one_minus", "reciprocal", "max_minus".
///
/// Returns
/// -------
/// (list[str], list[list[float]])
///   Tuple with the term order and a square distance matrix.
#[pyfunction]
#[pyo3(signature = (terms, method="lin", counter=None, distance_transform="auto"))]
pub fn term_distance_matrix(
    terms: Vec<String>,
    method: &str,
    counter: Option<&TermCounter>,
    distance_transform: &str,
) -> PyResult<(Vec<String>, Vec<Vec<f64>>)> {
    let (ordered_terms, mut matrix) = term_similarity_matrix(terms, method, counter)?;
    if ordered_terms.is_empty() {
        return Ok((ordered_terms, matrix));
    }

    let is_normalized = matches!(
        method.to_ascii_lowercase().as_str(),
        "lin" | "wang" | "simrel" | "topoicsim"
    );
    let mode = match distance_transform.to_ascii_lowercase().as_str() {
        "auto" => {
            if is_normalized {
                "one_minus"
            } else {
                "max_minus"
            }
        }
        "one_minus" | "one-minus" | "1-sim" | "1_minus" => "one_minus",
        "reciprocal" | "inv" | "inverse" => "reciprocal",
        "max_minus" | "max-minus" | "max" => "max_minus",
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown distance_transform '{}'. Options: auto, one_minus, reciprocal, max_minus",
                distance_transform
            )))
        }
    };

    match mode {
        "max_minus" => {
            let mut max_sim = 0.0_f64;
            for row in &matrix {
                for &v in row {
                    if v.is_finite() && v > max_sim {
                        max_sim = v;
                    }
                }
            }
            matrix.par_iter_mut().for_each(|row| {
                for v in row.iter_mut() {
                    let d = if v.is_finite() { max_sim - *v } else { 0.0 };
                    *v = if d < 0.0 { 0.0 } else { d };
                }
            });
        }
        "one_minus" => {
            matrix.par_iter_mut().for_each(|row| {
                for v in row.iter_mut() {
                    let d = 1.0 - *v;
                    *v = if d < 0.0 { 0.0 } else { d };
                }
            });
        }
        "reciprocal" => {
            matrix.par_iter_mut().for_each(|row| {
                for v in row.iter_mut() {
                    *v = if v.is_infinite() { 0.0 } else { 1.0 / (1.0 + *v) };
                }
            });
        }
        _ => unreachable!(),
    }

    for (i, row) in matrix.iter_mut().enumerate() {
        row[i] = 0.0;
    }

    Ok((ordered_terms, matrix))
}

/// Group semantically redundant GO terms into clusters around their most informative representative.
///
/// Arguments
/// ---------
/// terms : list of str
///   Input GO term IDs.
/// threshold : float
///   Similarity threshold above or equal to which two terms are considered redundant (default: 0.7).
/// method : str
///   Pairwise semantic similarity method (default: "lin").
/// counter : TermCounter
///   Precomputed IC values.
/// scores : Optional[list[float]]
///   Optional priority score per term (higher = preferred representative). If None, Information Content (IC) is used.
///
/// Returns
/// -------
/// dict[str, list[str]]
///   Mapping from each representative GO term ID to the list of GO terms in its cluster (including itself).
#[pyfunction]
#[pyo3(signature = (terms, threshold=0.7, method="lin", counter=None, scores=None))]
pub fn cluster_go_terms(
    terms: Vec<String>,
    threshold: f64,
    method: &str,
    counter: Option<&TermCounter>,
    scores: Option<Vec<f64>>,
) -> PyResult<std::collections::HashMap<String, Vec<String>>> {
    let counter_ref =
        counter.ok_or_else(|| PyValueError::new_err("counter argument is required"))?;
    if threshold <= 0.0 {
        return Err(PyValueError::new_err("threshold must be > 0.0"));
    }
    if let Some(ref sc) = scores {
        if sc.len() != terms.len() {
            return Err(PyValueError::new_err(
                "scores length must match terms length",
            ));
        }
    }

    // Deduplicate while preserving first-seen index
    let mut unique_terms: Vec<String> = Vec::with_capacity(terms.len());
    let mut unique_scores: Vec<f64> = Vec::with_capacity(terms.len());
    let mut seen: HashSet<String> = HashSet::default();
    for (idx, t) in terms.into_iter().enumerate() {
        if seen.insert(t.clone()) {
            let priority = match scores {
                Some(ref sc) => sc[idx],
                None => term_ic(&t, counter_ref),
            };
            unique_terms.push(t);
            unique_scores.push(priority);
        }
    }

    let n = unique_terms.len();
    if n == 0 {
        return Ok(std::collections::HashMap::new());
    }

    let (_, sim_matrix) = term_similarity_matrix(unique_terms.clone(), method, Some(counter_ref))?;

    // Order indices by descending priority score (ties broken by higher IC, then ID)
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        unique_scores[b]
            .partial_cmp(&unique_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                term_ic(&unique_terms[b], counter_ref)
                    .partial_cmp(&term_ic(&unique_terms[a], counter_ref))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| unique_terms[a].cmp(&unique_terms[b]))
    });

    let mut assigned = vec![false; n];
    let mut clusters: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();

    for &i in &order {
        if assigned[i] {
            continue;
        }
        assigned[i] = true;
        let mut members = vec![unique_terms[i].clone()];
        for &j in &order {
            if !assigned[j] && sim_matrix[i][j] >= threshold {
                assigned[j] = true;
                members.push(unique_terms[j].clone());
            }
        }
        clusters.insert(unique_terms[i].clone(), members);
    }

    Ok(clusters)
}

/// Reduce a list of GO terms by removing semantically redundant terms above a similarity threshold.
///
/// Arguments
/// ---------
/// terms : list of str
///   Input GO term IDs.
/// threshold : float
///   Similarity threshold above or equal to which two terms are considered redundant (default: 0.7).
/// method : str
///   Pairwise semantic similarity method (default: "lin").
/// counter : TermCounter
///   Precomputed IC values.
/// scores : Optional[list[float]]
///   Optional priority score per term (higher = preferred representative). If None, Information Content (IC) is used.
///
/// Returns
/// -------
/// list of str
///   Non-redundant representative GO term IDs in their original relative order.
#[pyfunction]
#[pyo3(signature = (terms, threshold=0.7, method="lin", counter=None, scores=None))]
pub fn reduce_redundant_terms(
    terms: Vec<String>,
    threshold: f64,
    method: &str,
    counter: Option<&TermCounter>,
    scores: Option<Vec<f64>>,
) -> PyResult<Vec<String>> {
    let clusters = cluster_go_terms(terms.clone(), threshold, method, counter, scores)?;
    let mut seen: HashSet<String> = HashSet::default();
    let mut representatives = Vec::with_capacity(clusters.len());
    for t in terms {
        if clusters.contains_key(&t) && seen.insert(t.clone()) {
            representatives.push(t);
        }
    }
    Ok(representatives)
}


