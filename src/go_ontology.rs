use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use pyo3::prelude::*;
use pyo3::types::PyString;
use crate::go_loader::{GO_TERMS_CACHE, GENE2GO_CACHE};
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
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Gene2GO mapping not loaded. Call go3.load_gene2go() first."))?
            .read()
    )
}

/// Get the PyGOTerm object for a given GO term ID.
///
/// Arguments
/// ---------
/// go_id : str
///   GO term ID.
///
/// Returns
/// -------
/// Option<PyGOTerm>
///   The GO term as a Python object, or None if not found.
#[pyfunction]
pub fn get_term_by_id(go_id: &str) -> PyResult<Option<PyGOTerm>> {
    let terms = get_terms_or_error()?;
    Ok(terms.get(go_id).map(PyGOTerm::from))
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
    visited.remove(go_id);
    visited
}

/// Returns the list of all ancestors in the ontology for the given GO Term.
///
/// Arguments
/// ---------
/// go_id : str
///   GO term ID.
///
/// Returns
/// -------
/// list of str
///   List of IDs of all the ancestors in the ontology.
#[pyfunction]
pub fn ancestors(go_id: &str) -> PyResult<Vec<String>> {
    let terms = get_terms_or_error()?;
    let visited = collect_ancestors(go_id, &terms);
    Ok(visited.into_iter().collect())
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
    let (id_a, id_b) = if go_id1 <= go_id2 {
        (go_id1, go_id2)
    } else {
        (go_id2, go_id1)
    };

    // Try to use the DCA cache if available
    if let Some(lock) = crate::go_loader::DCA_CACHE.get() {
        let cache = lock.write();
        if let Some(result) = cache.get(&(id_a.to_string(), id_b.to_string())) {
            return Ok(Some(result.clone()));
        }
    }

    let set1 = collect_ancestors(id_a, &terms);
    let set2 = collect_ancestors(id_b, &terms);
    let mut best = None;
    let mut max_depth = 0;
    for term_id in set1.intersection(&set2) {
        if let Some(term) = terms.get(term_id) {
            if let Some(depth) = term.depth {
                if depth >= max_depth {
                    max_depth = depth;
                    best = Some(term_id.to_string());
                }
            }
        }
    }

    // Store the result in the cache if available
    if let Some(lock) = crate::go_loader::DCA_CACHE.get() {
        let mut cache = lock.write();
        if let Some(ref dca) = best {
            cache.insert((id_a.to_string(), id_b.to_string()), dca.clone());
        }
    }

    Ok(best)
}