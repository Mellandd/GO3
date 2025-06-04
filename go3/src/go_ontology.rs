use std::collections::{HashMap, HashSet};
use pyo3::prelude::*;
use pyo3::types::PyString;
use crate::go_loader::GO_TERMS_CACHE;

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

pub fn get_terms_or_error<'a>() -> PyResult<std::sync::RwLockReadGuard<'a, HashMap<String, GOTerm>>> {
    GO_TERMS_CACHE
        .get()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("GO terms not loaded. Call go3.load_go_terms() first."))?
        .read()
        .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("Failed to read GO terms"))
}

/// Gets the PyGoTerm object for the given GO Term ID.
///
/// # Arguments
///
/// * `go_id` - GO term ID.
///
/// # Returns
///
/// PyGOTerm associated with the ID. (PyGOTerm)
#[pyfunction]
pub fn get_term_by_id(go_id: &str) -> PyResult<Option<PyGOTerm>> {
    let terms = get_terms_or_error()?;
    Ok(terms.get(go_id).map(PyGOTerm::from))
}

pub fn collect_ancestors<'a>(go_id: &'a str, terms: &'a HashMap<String, GOTerm>) -> HashSet<&'a str> {
    let mut visited = HashSet::new();
    let mut stack = vec![go_id];
    while let Some(current) = stack.pop() {
        if visited.insert(current) {
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
/// # Arguments
///
/// * `go_id` - GO term ID.
///
/// # Returns
///
/// List of IDs of all the ancestors in the ontology (List of String)
#[pyfunction]
pub fn ancestors(go_id: &str) -> PyResult<Vec<String>> {
    let terms = get_terms_or_error()?;
    let visited = collect_ancestors(go_id, &terms);
    Ok(visited.into_iter().map(str::to_string).collect())
}

/// Returns the list of all the common ancestors in the ontology for the given GO Terms.
///
/// # Arguments
///
/// * `go_id1` - GO term ID 1.
/// * `go_id2` - GO term ID 2.
///
/// # Returns
///
/// List of IDs of all the common ancestors in the ontology (List of String)
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
/// # Arguments
///
/// * `go_id1` - GO term ID 1.
/// * `go_id2` - GO term ID 2.
///
/// # Returns
///
/// ID of the deepest common ancestor in the ontology. (String)
#[pyfunction]
pub fn deepest_common_ancestor(go_id1: &str, go_id2: &str) -> PyResult<Option<String>> {
    let terms = get_terms_or_error()?;
    let set1 = collect_ancestors(go_id1, &terms);
    let set2 = collect_ancestors(go_id2, &terms);
    let mut best = None;
    let mut max_depth = 0;
    for &term_id in set1.intersection(&set2) {
        if let Some(term) = terms.get(term_id) {
            if let Some(depth) = term.depth {
                if depth >= max_depth {
                    max_depth = depth;
                    best = Some(term_id.to_string());
                }
            }
        }
    }
    Ok(best)
}