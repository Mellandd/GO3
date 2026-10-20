use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use pyo3::prelude::*;
use pyo3::types::PyString;
use crate::go_loader::{GO_TERMS_CACHE, GENE2GO_CACHE};
use pyo3::exceptions::PyValueError;

/// Struct representing a Gene Ontology (GO) term.
///
/// Fields
/// ------
/// id : str
///   GO term identifier (e.g., GO:0006397).
/// name : str
///   Human-readable name of the term.
/// namespace : str
///   Ontology namespace (e.g., biological_process).
/// definition : str
///   Textual definition of the term.
/// parents : list of str
///   List of parent GO term IDs (is_a relationships).
/// children : list of str
///   List of child GO term IDs (is_a relationships).
/// depth : Optional[int]
///   Maximum distance to a root term (None if not computed).
/// level : Optional[int]
///   Minimum distance to a root term (None if not computed).
/// is_obsolete : bool
///   True if the term is obsolete.
/// alt_ids : list of str
///   Alternative GO IDs for this term.
/// replaced_by : Optional[str]
///   If obsolete, the term that replaces this one.
/// consider : list of str
///   Suggested replacement terms if obsolete.
/// synonyms : list of str
///   List of synonyms.
/// xrefs : list of str
///   Cross-references to other databases.
/// relationships : list of (str, str)
///   Other relationships (e.g., part_of).
/// comment : Optional[str]
///   Additional comments.
#[derive(Clone)]
pub struct GOTerm {
    pub id: String,
    pub name: String,
    pub namespace: String,
    pub definition: String,
    pub parents: Vec<String>,
    pub children: Vec<String>,
    pub depth: Option<usize>,
    pub level: Option<usize>,
    pub is_obsolete: bool,
    pub alt_ids: Vec<String>,
    pub replaced_by: Option<String>,
    pub consider: Vec<String>,
    pub synonyms: Vec<String>,
    pub xrefs: Vec<String>,
    pub relationships: Vec<(String, String)>,
    pub comment: Option<String>,
}

/// Python-exposed struct representing a GO term (for use in the Python API).
///
/// Fields
/// ------
/// id : str
///   GO term identifier.
/// name : str
///   Human-readable name of the term.
/// namespace : str
///   Ontology namespace.
/// definition : str
///   Textual definition of the term.
/// parents : list of str
///   List of parent GO term IDs.
/// children : list of str
///   List of child GO term IDs.
/// depth : Optional[int]
///   Maximum distance to a root term.
/// level : Optional[int]
///   Minimum distance to a root term.
/// is_obsolete : bool
///   True if the term is obsolete.
/// alt_ids : list of str
///   Alternative GO IDs for this term.
/// replaced_by : Optional[str]
///   If obsolete, the term that replaces this one.
/// consider : list of str
///   Suggested replacement terms if obsolete.
/// synonyms : list of str
///   List of synonyms.
/// xrefs : list of str
///   Cross-references to other databases.
/// relationships : list of (str, str)
///   Other relationships (e.g., part_of).
/// comment : Optional[str]
///   Additional comments.
#[pyclass]
#[derive(Clone)]
pub struct PyGOTerm {
    #[pyo3(get)] pub id: String,
    #[pyo3(get)] pub name: String,
    #[pyo3(get)] pub namespace: String,
    #[pyo3(get)] pub definition: String,
    #[pyo3(get)] pub parents: Vec<String>,
    #[pyo3(get)] pub children: Vec<String>,
    #[pyo3(get)] pub depth: Option<usize>,
    #[pyo3(get)] pub level: Option<usize>,
    #[pyo3(get)] pub is_obsolete: bool,
    #[pyo3(get)] pub alt_ids: Vec<String>,
    #[pyo3(get)] pub replaced_by: Option<String>,
    #[pyo3(get)] pub consider: Vec<String>,
    #[pyo3(get)] pub synonyms: Vec<String>,
    #[pyo3(get)] pub xrefs: Vec<String>,
    #[pyo3(get)] pub relationships: Vec<(String, String)>,
    #[pyo3(get)] pub comment: Option<String>,
}

impl From<&GOTerm> for PyGOTerm {
    fn from(term: &GOTerm) -> Self {
        Self {
            id: term.id.clone(),
            name: term.name.clone(),
            namespace: term.namespace.clone(),
            definition: term.definition.clone(),
            parents: term.parents.clone(),
            children: term.children.clone(),
            depth: term.depth,
            level: term.level,
            is_obsolete: term.is_obsolete,
            alt_ids: term.alt_ids.clone(),
            replaced_by: term.replaced_by.clone(),
            consider: term.consider.clone(),
            synonyms: term.synonyms.clone(),
            xrefs: term.xrefs.clone(),
            relationships: term.relationships.clone(),
            comment: term.comment.clone(),
        }
    }
}

#[pymethods]
impl PyGOTerm {
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let class_name: Bound<'_, PyString> = slf.get_type().qualname()?;
        let s = slf.borrow();
        Ok(format!(
            "{} id: {}\nname: {}\nnamespace: {}\ndefinition: {}\nparents: {:?}\nchildren: {:?}\ndepth: {:?}\nlevel: {:?}\nis_obsolete: {}\nalt_ids: {:?}\nreplaced_by: {:?}\nconsider: {:?}\nsynonyms: {:?}\nxrefs: {:?}\nrelationships: {:?}\ncomments: {:?}",
            class_name, s.id, s.name, s.namespace, s.definition, s.parents, s.children, s.depth, s.level,
            s.is_obsolete, s.alt_ids, s.replaced_by, s.consider, s.synonyms, s.xrefs, s.relationships, s.comment
        ))
    }
}

impl From<PyGOTerm> for GOTerm {
    fn from(py_term: PyGOTerm) -> Self {
        Self {
            id: py_term.id,
            name: py_term.name,
            namespace: py_term.namespace,
            definition: py_term.definition,
            parents: py_term.parents,
            children: py_term.children,
            depth: py_term.depth,
            level: py_term.level,
            is_obsolete: py_term.is_obsolete,
            alt_ids: py_term.alt_ids,
            replaced_by: py_term.replaced_by,
            consider: py_term.consider,
            synonyms: py_term.synonyms,
            xrefs: py_term.xrefs,
            relationships: py_term.relationships,
            comment: py_term.comment,
        }
    }
}

/// Get a read lock on the global GO terms map, or error if not loaded.
///
/// Arguments
/// ---------
/// None
///
/// Returns
/// -------
/// RwLockReadGuard<HashMap<String, GOTerm>>
///   Read guard for the GO terms map.
pub fn get_terms_or_error<'a>() -> PyResult<parking_lot::RwLockReadGuard<'a, HashMap<String, GOTerm>>> {
    Ok(
        GO_TERMS_CACHE
            .get()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("GO terms not loaded. Call go3.load_go_terms() first."))?
            .read()
    )
}

/// Get a read lock on the global gene-to-GO mapping, or error if not loaded.
///
/// Arguments
/// ---------
/// None
///
/// Returns
/// -------
/// RwLockReadGuard<HashMap<String, Vec<String>>>
///   Read guard for the gene2go map.
pub fn get_gene2go_or_error<'a>() -> PyResult<parking_lot::RwLockReadGuard<'a, HashMap<String, Vec<String>>>> {
    Ok(
        GENE2GO_CACHE
            .get()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Gene2GO mapping not loaded. Call go3.load_gaf() first."))?
            .read()
    )
}


/// Get the PyGOTerm object for a given GO term ID.
///
/// Raises:
///     ValueError: If the GO term does not exist in the ontology.
#[pyfunction]
pub fn get_term_by_id(go_id: &str) -> PyResult<PyGOTerm> {
    let terms = get_terms_or_error()?;
    match terms.get(go_id) {
        Some(term) => Ok(PyGOTerm::from(term)),
        None => Err(PyValueError::new_err(format!(
            "GO term '{}' not found in ontology",
            go_id
        ))),
    }
}

/// Collect all ancestors of a GO term (recursively via is_a).
///
/// Arguments
/// ---------
/// go_id : str
///   GO term ID.
/// terms : dict
///   Map of GO terms.
///
/// Returns
/// -------
/// HashSet<String>
///   Set of ancestor GO term IDs.
pub fn collect_ancestors(go_id: &str, terms: &HashMap<String, GOTerm>) -> HashSet<String> {
    // Try to use the precomputed cache if available
    if let Some(lock) = crate::go_loader::ANCESTORS_CACHE.get() {
        let cache = lock.read();
        if let Some(ancestors) = cache.get(go_id) {
            return ancestors.clone();
        }
    }
    // Fallback: original computation
    let mut visited = HashSet::default();
    let mut stack = vec![go_id];
    while let Some(current) = stack.pop() {
        if visited.insert(current.to_string()) {
            if let Some(term) = terms.get(current) {
                for parent in &term.parents {
                    stack.push(parent);
                }
            }
        }
    }
//    visited.remove(go_id);
    visited
}

/// Returns the list of all ancestors in the ontology for the given GO Term.
///
/// Arguments
/// ---------
/// go_id : str
///   GO term ID.
/// relations : Optional[Iterable[str]]
///   Optional additional relationship types to traverse besides `is_a` (e.g., `["part_of"]`).
///
/// Returns
/// -------
/// list of str
///   List of IDs of all the ancestors in the ontology.
#[pyfunction]
#[pyo3(signature = (go_id, relations=None))]
pub fn ancestors(go_id: &str, relations: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<String>> {
    let terms = get_terms_or_error()?;
    let Some(rel_obj) = relations else {
        let visited = collect_ancestors(go_id, &terms);
        return Ok(visited.into_iter().collect());
    };

    let mut rel_set: HashSet<String> = HashSet::default();
    for item in rel_obj.try_iter()? {
        let s: String = item?.extract()?;
        rel_set.insert(s.trim().to_string());
    }

    if rel_set.is_empty() || (rel_set.len() == 1 && rel_set.contains("is_a")) {
        let visited = collect_ancestors(go_id, &terms);
        return Ok(visited.into_iter().collect());
    }

    let mut visited = HashSet::default();
    let mut stack = vec![go_id.to_string()];
    while let Some(current) = stack.pop() {
        if visited.insert(current.clone()) {
            if let Some(term) = terms.get(&current) {
                for parent in &term.parents {
                    stack.push(parent.clone());
                }
                for (rel_type, target) in &term.relationships {
                    if rel_set.contains(rel_type) {
                        stack.push(target.clone());
                    }
                }
            }
        }
    }
    Ok(visited.into_iter().collect())
}

/// Returns the list of all descendants in the ontology for the given GO Term.
///
/// Arguments
/// ---------
/// go_id : str
///   GO term ID.
/// relations : Optional[Iterable[str]]
///   Optional additional relationship types to traverse besides `is_a` (e.g., `["part_of"]`).
///
/// Returns
/// -------
/// list of str
///   Sorted list of IDs of all the descendants in the ontology (including `go_id`).
#[pyfunction]
#[pyo3(signature = (go_id, relations=None))]
pub fn descendants(go_id: &str, relations: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<String>> {
    let terms = get_terms_or_error()?;
    if !terms.contains_key(go_id) {
        return Err(PyValueError::new_err(format!(
            "GO term '{}' not found in ontology",
            go_id
        )));
    }

    let mut rel_set: HashSet<String> = HashSet::default();
    if let Some(rel_obj) = relations {
        for item in rel_obj.try_iter()? {
            let s: String = item?.extract()?;
            rel_set.insert(s.trim().to_string());
        }
        rel_set.remove("is_a");
    }

    // Build reverse map for extra relationships if requested
    let mut extra_children: HashMap<&str, Vec<&str>> = HashMap::default();
    if !rel_set.is_empty() {
        for (id, term) in terms.iter() {
            if id != &term.id {
                continue; // skip alt_id duplicates
            }
            for (rel_type, target) in &term.relationships {
                if rel_set.contains(rel_type) {
                    extra_children.entry(target.as_str()).or_default().push(id.as_str());
                }
            }
        }
    }

    let mut visited: HashSet<String> = HashSet::default();
    let mut stack = vec![go_id.to_string()];
    while let Some(current) = stack.pop() {
        if visited.insert(current.clone()) {
            if let Some(term) = terms.get(&current) {
                for child in &term.children {
                    stack.push(child.clone());
                }
            }
            if let Some(extra) = extra_children.get(current.as_str()) {
                for &child in extra {
                    stack.push(child.to_string());
                }
            }
        }
    }

    let mut result: Vec<String> = visited.into_iter().collect();
    result.sort_unstable();
    Ok(result)
}

/// Returns the list of all the common ancestors in the ontology for the given GO Terms.
///
/// Arguments
/// ---------
/// go_id1 : str
///   GO term ID 1.
/// go_id2 : str
///   GO term ID 2.
///
/// Returns
/// -------
/// list of str
///   List of IDs of all the common ancestors in the ontology.
#[pyfunction]
pub fn common_ancestor(go_id1: &str, go_id2: &str) -> PyResult<Vec<String>> {
    let terms = get_terms_or_error()?;
    let set1 = collect_ancestors(go_id1, &terms);
    let set2 = collect_ancestors(go_id2, &terms);
    let mut common: Vec<String> = set1.intersection(&set2).map(|s| (*s).to_string()).collect();
    common.sort_unstable();
    Ok(common)
}

/// Returns the deepest common ancestor in the ontology for the given GO Terms.
///
/// Arguments
/// ---------
/// go_id1 : str
///   GO term ID 1.
/// go_id2 : str
///   GO term ID 2.
///
/// Returns
/// -------
/// Option<String>
///   ID of the deepest common ancestor in the ontology.
#[pyfunction]
pub fn deepest_common_ancestor(go_id1: &str, go_id2: &str) -> PyResult<Option<String>> {
    let terms = get_terms_or_error()?;

    if !terms.contains_key(go_id1) {
        return Err(PyValueError::new_err(format!(
            "GO term '{}' not found in ontology",
            go_id1
        )));
    }
    if !terms.contains_key(go_id2) {
        return Err(PyValueError::new_err(format!(
            "GO term '{}' not found in ontology",
            go_id2
        )));
    }

    let (id_a, id_b) = if go_id1 <= go_id2 {
        (go_id1, go_id2)
    } else {
        (go_id2, go_id1)
    };

    // Try to use the DCA cache if available
    let cache_key = (id_a.to_string(), id_b.to_string());
    if let Some(lock) = crate::go_loader::DCA_CACHE.get() {
        let cache = lock.read();
        if let Some(result) = cache.get(&cache_key) {
            return Ok(Some(result.clone()));
        }
    }

    // Fast path: if we have the precomputed ancestors cache, avoid cloning large HashSets
    // on every DCA query by intersecting the cached sets by reference.
    let best = if let Some(lock) = crate::go_loader::ANCESTORS_CACHE.get() {
        let cache = lock.read();
        match (cache.get(id_a), cache.get(id_b)) {
            (Some(set1), Some(set2)) => {
                let mut best = None;
                let mut max_depth = 0;
                for term_id in set1.intersection(set2) {
                    if let Some(term) = terms.get(term_id.as_str()) {
                        if let Some(depth) = term.depth {
                            if depth >= max_depth {
                                max_depth = depth;
                                best = Some(term_id.clone());
                            }
                        }
                    }
                }
                best
            }
            _ => {
                // Fallback: compute ancestors (may still use cache internally).
                let set1 = collect_ancestors(id_a, &terms);
                let set2 = collect_ancestors(id_b, &terms);
                let mut best = None;
                let mut max_depth = 0;
                for term_id in set1.intersection(&set2) {
                    if let Some(term) = terms.get(term_id.as_str()) {
                        if let Some(depth) = term.depth {
                            if depth >= max_depth {
                                max_depth = depth;
                                best = Some(term_id.clone());
                            }
                        }
                    }
                }
                best
            }
        }
    } else {
        let set1 = collect_ancestors(id_a, &terms);
        let set2 = collect_ancestors(id_b, &terms);
        let mut best = None;
        let mut max_depth = 0;
        for term_id in set1.intersection(&set2) {
            if let Some(term) = terms.get(term_id.as_str()) {
                if let Some(depth) = term.depth {
                    if depth >= max_depth {
                        max_depth = depth;
                        best = Some(term_id.clone());
                    }
                }
            }
        }
        best
    };

    // Store the result in the cache if available
    if let (Some(lock), Some(dca)) = (crate::go_loader::DCA_CACHE.get(), best.as_ref()) {
        let mut cache = lock.write();
        cache.insert(cache_key, dca.clone());
    }

    Ok(best)
}

fn resolve_ontology_namespace(ontology: &str) -> PyResult<&'static str> {
    match ontology.trim().to_ascii_uppercase().as_str() {
        "BP" | "P" | "BIOLOGICAL_PROCESS" => Ok("biological_process"),
        "MF" | "F" | "MOLECULAR_FUNCTION" => Ok("molecular_function"),
        "CC" | "C" | "CELLULAR_COMPONENT" => Ok("cellular_component"),
        _ => Err(PyValueError::new_err(format!(
            "Invalid ontology '{}'. Must be 'BP', 'MF', or 'CC'",
            ontology
        ))),
    }
}

/// Return a sorted list of all loaded gene symbols (optionally filtered by subontology).
#[pyfunction]
#[pyo3(signature = (ontology=None))]
pub fn list_genes(ontology: Option<&str>) -> PyResult<Vec<String>> {
    let gene2go = get_gene2go_or_error()?;
    let mut genes: Vec<String> = match ontology {
        None => gene2go.keys().cloned().collect(),
        Some(ont) => {
            let ns = resolve_ontology_namespace(ont)?;
            let terms = get_terms_or_error()?;
            gene2go
                .iter()
                .filter(|(_, go_ids)| {
                    go_ids.iter().any(|go_id| {
                        terms
                            .get(go_id.as_str())
                            .map_or(false, |t| t.namespace.eq_ignore_ascii_case(ns))
                    })
                })
                .map(|(gene, _)| gene.clone())
                .collect()
        }
    };
    genes.sort_unstable();
    Ok(genes)
}

/// Return the unique sorted GO term IDs annotated to a given gene.
///
/// Arguments
/// ---------
/// gene : str
///   Gene symbol.
/// ontology : Optional[str]
///   Optional subontology filter ("BP", "MF", or "CC").
#[pyfunction]
#[pyo3(signature = (gene, ontology=None))]
pub fn get_terms_for_gene(gene: &str, ontology: Option<&str>) -> PyResult<Vec<String>> {
    let gene2go = get_gene2go_or_error()?;
    let raw_terms = gene2go.get(gene).ok_or_else(|| {
        PyValueError::new_err(format!("Gene '{}' not found in mapping", gene))
    })?;

    let mut result: Vec<String> = match ontology {
        None => raw_terms.clone(),
        Some(ont) => {
            let ns = resolve_ontology_namespace(ont)?;
            let terms = get_terms_or_error()?;
            raw_terms
                .iter()
                .filter(|go_id| {
                    terms
                        .get(go_id.as_str())
                        .map_or(false, |t| t.namespace.eq_ignore_ascii_case(ns))
                })
                .cloned()
                .collect()
        }
    };
    result.sort_unstable();
    result.dedup();
    Ok(result)
}

/// Return all gene symbols annotated to the given GO term (optionally including descendant terms).
///
/// Arguments
/// ---------
/// go_id : str
///   GO term identifier.
/// include_descendants : bool
///   If True (default), includes genes annotated to any descendant of `go_id`.
#[pyfunction]
#[pyo3(signature = (go_id, include_descendants=true))]
pub fn get_genes_for_term(go_id: &str, include_descendants: bool) -> PyResult<Vec<String>> {
    let terms = get_terms_or_error()?;
    let target_term = terms.get(go_id).ok_or_else(|| {
        PyValueError::new_err(format!("GO term '{}' not found in ontology", go_id))
    })?;
    let target_ids: HashSet<&str> = target_term
        .alt_ids
        .iter()
        .map(String::as_str)
        .chain(std::iter::once(go_id))
        .collect();

    let gene2go = get_gene2go_or_error()?;
    let ancestors_guard = crate::go_loader::ANCESTORS_CACHE.get().map(|l| l.read());
    let ancestors_cache = ancestors_guard.as_deref();

    let mut matched_genes: Vec<String> = gene2go
        .iter()
        .filter(|(_, go_ids)| {
            if !include_descendants {
                go_ids.iter().any(|id| target_ids.contains(id.as_str()))
            } else {
                go_ids.iter().any(|id| {
                    if target_ids.contains(id.as_str()) {
                        return true;
                    }
                    if let Some(cache) = ancestors_cache {
                        if let Some(ancs) = cache.get(id.as_str()) {
                            return target_ids.iter().any(|tid| ancs.contains(*tid));
                        }
                    }
                    let ancs = collect_ancestors(id, &terms);
                    target_ids.iter().any(|tid| ancs.contains(*tid))
                })
            }
        })
        .map(|(gene, _)| gene.clone())
        .collect();

    matched_genes.sort_unstable();
    Ok(matched_genes)
}

