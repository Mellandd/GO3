use pyo3::prelude::*;
use once_cell::sync::OnceCell;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::io::{BufReader, BufRead};
use std::fs::File;
use parking_lot::RwLock;
use std::path::Path;
use std::fs;
use reqwest::blocking::get;
use rayon::prelude::*;

use crate::go_ontology::{GOTerm, PyGOTerm, collect_ancestors, get_terms_or_error};
pub static GO_TERMS_CACHE: OnceCell<RwLock<HashMap<String, GOTerm>>> = OnceCell::new();
pub static GENE2GO_CACHE: OnceCell<RwLock<HashMap<String, Vec<String>>>> = OnceCell::new();
pub static ANCESTORS_CACHE: OnceCell<RwLock<HashMap<String, HashSet<String>>>> = OnceCell::new();
pub static DCA_CACHE: OnceCell<RwLock<HashMap<(String, String), String>>> = OnceCell::new();

/// Struct representing a single annotation from a GAF file.
///
/// Fields
/// ------
/// db_object_id : str
///   The gene product identifier (e.g., UniProt ID).
/// go_term : str
///   The GO term ID (e.g., GO:0008150).
/// evidence : str
///   The evidence code for the annotation (e.g., IEA).
#[pyclass]
#[derive(Clone)]
pub struct GAFAnnotation {
    #[pyo3(get)]
    pub db_object_id: String,
    #[pyo3(get)]
    pub go_term: String,
    #[pyo3(get)]
    pub evidence: String,
}

/// Struct holding annotation counts and information content (IC) for GO terms.
///
/// Fields
/// ------
/// counts : dict
///   Mapping from GO term ID to annotation count.
/// total_by_ns : dict
///   Mapping from namespace to total annotation count.
/// ic : dict
///   Mapping from GO term ID to information content (IC).
#[pyclass]
#[derive(Clone)]
pub struct TermCounter {
    #[pyo3(get)]
    pub counts: HashMap<String, usize>,         // term_id -> count
    #[pyo3(get)]
    pub total_by_ns: HashMap<String, usize>,    // namespace -> total annotations
    #[pyo3(get)]
    pub ic: HashMap<String, f64>,               // term_id -> IC
}

/// Parse a GO OBO file and return a map of GO term IDs to GOTerm structs.
///
/// Arguments
/// ---------
/// path : str
///   Path to the OBO file.
///
/// Returns
/// -------
/// dict
///   Map of GO term IDs to term structs.
pub fn parse_obo(path: &str) -> HashMap<String, GOTerm> {
    let contents = fs::read_to_string(path).expect("Can't open OBO file");
    let chunks = contents.split("[Term]");

    let canonical_terms: Vec<GOTerm> = chunks
        .par_bridge()
        .filter_map(parse_term_chunk)
        .filter(|term| !term.is_obsolete)
        .collect();

    let mut term_map: HashMap<String, GOTerm> =
        HashMap::with_capacity_and_hasher(canonical_terms.len() * 2, Default::default());

    for term in canonical_terms.into_iter() {
        // Collect the "synonym group": canonical + alts
        let mut all_ids = term.alt_ids.clone();
        all_ids.push(term.id.clone());

        // Build GOTerm copies for each ID
        for id in &all_ids {
            let mut clone = term.clone();
            clone.id = id.clone();
            clone.alt_ids = all_ids.clone(); // all point to the same group
            term_map.insert(id.clone(), clone);
        }
    }

    // Compute children, levels, depths, etc.
    compute_levels_and_depths(&mut term_map);

    // Precompute ancestors for caching
    let ancestors_map: HashMap<String, HashSet<String>> = term_map
        .par_iter()
        .map(|(id, _)| {
            let ancestors = crate::go_ontology::collect_ancestors(id, &term_map)
                .into_iter()
                .map(|s| s.to_string())
                .collect();
            (id.clone(), ancestors)
        })
        .collect();
    let _ = ANCESTORS_CACHE.set(RwLock::new(ancestors_map));

    // Initialize DCA_CACHE
    let _ = DCA_CACHE.set(RwLock::new(HashMap::default()));

    term_map
}

/// Parse a chunk of text representing a single GO term from an OBO file.
///
/// Arguments
/// ---------
/// chunk : str
///   Text chunk for a single term.
///
/// Returns
/// -------
/// Option[GOTerm]
///   Parsed term or None if invalid/obsolete.
fn parse_term_chunk(chunk: &str) -> Option<GOTerm> {
    let mut term = GOTerm {
        id: String::new(),
        name: String::new(),
        namespace: String::new(),
        definition: String::new(),
        parents: Vec::new(),
        is_obsolete: false,
        alt_ids: Vec::new(),
        replaced_by: None,
        consider: Vec::new(),
        synonyms: Vec::new(),
        xrefs: Vec::new(),
        relationships: Vec::new(),
        comment: None,
        children: Vec::new(),
        level: None,
        depth: None,
    };

    let chunk = chunk.split("[Typedef]").next().unwrap_or(chunk);

    let lines: Vec<&str> = chunk
        .lines()
        .map(|l| l.trim()) // eliminamos espacios a izquierda y derecha
        .filter(|l| !l.is_empty()) // quitamos líneas vacías
        .collect();

    if lines.is_empty() {
        return None;
    }
    let mut valid = false;

    for line in lines {
        if line.starts_with("id: ") {
            term.id = line["id: ".len()..].to_string();
            valid = true;
        } else if line.starts_with("name: ") {
            term.name = line["name: ".len()..].to_string();
        } else if line.starts_with("namespace: ") {
            term.namespace = line["namespace: ".len()..].to_string();
        } else if line.starts_with("def: ") {
            term.definition = line["def: ".len()..].to_string();
        } else if line.starts_with("is_a: ") {
            let parent = line["is_a: ".len()..]
                .split_whitespace()
                .next()
                .unwrap_or("")
                .to_string();
            if !parent.is_empty() {
                term.parents.push(parent);
            }
        } else if line.starts_with("alt_id: ") {
            term.alt_ids.push(line["alt_id: ".len()..].to_string());
        } else if line.starts_with("is_obsolete: true") {
            term.is_obsolete = true;
        } else if line.starts_with("replaced_by: ") {
            term.replaced_by = Some(line["replaced_by: ".len()..].to_string());
        } else if line.starts_with("consider: ") {
            term.consider.push(line["consider: ".len()..].to_string());
        } else if line.starts_with("synonym: ") {
            term.synonyms.push(line["synonym: ".len()..].to_string());
        } else if line.starts_with("xref: ") {
            term.xrefs.push(line["xref: ".len()..].to_string());
        } else if line.starts_with("relationship: ") {
            let rel_def = &line["relationship: ".len()..];
            let mut parts = rel_def.split_whitespace();
            if let (Some(rel), Some(target)) = (parts.next(), parts.next()) {
                term.relationships.push((rel.to_string(), target.to_string()));
            }
        }
    }

    if valid {
        Some(term)
    } else {
        None
    }
}

/// Compute the level and depth for each GO term in the ontology.
///
/// Arguments
/// ---------
/// terms : dict
///   Mutable map of GO terms.
///
/// Returns
/// -------
/// None
///   Updates the `level` and `depth` fields of each term in-place.
pub fn compute_levels_and_depths(terms: &mut HashMap<String, GOTerm>) {
    // Paso 1: construir mapa de hijos
    let mut child_map: HashMap<String, Vec<String>> = HashMap::default();
    for (id, term) in terms.iter() {
        for parent in &term.parents {
            child_map.entry(parent.clone()).or_default().push(id.clone());
        }
    }

    // Paso 2: inicializar level
    fn init_level(
        term_id: &str,
        terms: &mut HashMap<String, GOTerm>,
        visiting: &mut HashSet<String>,
    ) -> usize {
        if visiting.contains(term_id) {
            // Ciclo detectado: se evita recursión infinita
            eprintln!("⚠️ Ciclo detectado en level: {}", term_id);
            return 0;
        }

        if let Some(level) = terms.get(term_id).and_then(|t| t.level) {
            return level;
        }

        visiting.insert(term_id.to_string());

        let parents = terms
            .get(term_id)
            .map(|t| t.parents.clone())
            .unwrap_or_default();

        let level = if parents.is_empty() {
            0
        } else {
            parents
                .iter()
                .map(|p| init_level(p, terms, visiting))
                .min()
                .unwrap_or(0) + 1
        };

        visiting.remove(term_id);
        if let Some(term) = terms.get_mut(term_id) {
            term.level = Some(level);
        }

        level
    }

    // Paso 3: inicializar depth
    fn init_depth(
        term_id: &str,
        terms: &mut HashMap<String, GOTerm>,
        visiting: &mut HashSet<String>,
    ) -> usize {
        if visiting.contains(term_id) {
            eprintln!("Ciclo detectado en depth: {}", term_id);
            return 0;
        }

        if let Some(depth) = terms.get(term_id).and_then(|t| t.depth) {
            return depth;
        }

        visiting.insert(term_id.to_string());

        let parents = terms
            .get(term_id)
            .map(|t| t.parents.clone())
            .unwrap_or_default();

        let depth = if parents.is_empty() {
            0
        } else {
            parents
                .iter()
                .map(|p| init_depth(p, terms, visiting))
                .max()
                .unwrap_or(0) + 1
        };

        visiting.remove(term_id);
        if let Some(term) = terms.get_mut(term_id) {
            term.depth = Some(depth);
        }

        depth
    }

    // Paso 4: recorrer todos los términos y calcular level + depth
    let ids: Vec<String> = terms.keys().cloned().collect();
    for id in &ids {
        let mut visiting = HashSet::default();
        init_level(id, terms, &mut visiting);

        let mut visiting = HashSet::default();
        init_depth(id, terms, &mut visiting);
    }

    // Paso 5: rellenar el campo children con los hijos (solo vía is_a)
    for (parent, children) in child_map {
        if let Some(term) = terms.get_mut(&parent) {
            term.children = children;
        }
    }
}

/// Download the latest GO OBO file if not present locally.
///
/// Arguments
/// ---------
/// None
///
/// Returns
/// -------
/// str
///   Path to the downloaded or existing OBO file.
pub fn download_obo() -> Result<String, String> {
    let obo_path = "go-basic.obo";
    if Path::new(obo_path).exists() {
        return Ok(obo_path.to_string());
    }

    let url = "http://purl.obolibrary.org/obo/go/go-basic.obo";
    println!("Descargando ontología desde: {}", url);
    let response = get(url).map_err(|e| e.to_string())?;

    let content = response.text().map_err(|e| e.to_string())?;
    fs::write(obo_path, content).map_err(|e| e.to_string())?;

    Ok(obo_path.to_string())
}

/// Load GO terms from an OBO file and cache them globally.
///
/// Arguments
/// ---------
/// path : Optional[str]
///   Optional path to the OBO file.
///
/// Returns
/// -------
/// list of PyGOTerm
///   List of GO terms as Python objects.
#[pyfunction]
#[pyo3(signature = (path=None))]
pub fn load_go_terms(path: Option<String>) -> PyResult<Vec<PyGOTerm>> {
    let path = path.unwrap_or_else(|| download_obo().unwrap());
    let terms_map = parse_obo(&path);

    // Guardar en la caché global
    let _ = GO_TERMS_CACHE.set(RwLock::new(terms_map.clone()));

    // Devolver lista de PyGOTerm
    let terms_vec = terms_map
        .into_iter()
        .map(|(_, v)| PyGOTerm::from(&v))
        .collect();

    Ok(terms_vec)
}

/// Load a GAF annotation file and cache the gene-to-GO mapping.
///
/// Arguments
/// ---------
/// path : str
///   Path to the GAF file.
///
/// Returns
/// -------
/// list of GAFAnnotation
///   List of parsed GAF annotations.
#[pyfunction]
pub fn load_gaf(path: String) -> PyResult<Vec<GAFAnnotation>> {
    let file = File::open(&path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let reader = BufReader::new(file);

    // Get the loaded GO terms to check for obsolete terms
    let terms = match crate::go_ontology::get_terms_or_error() {
        Ok(t) => t,
        Err(e) => return Err(e),
    };

    let mut annotations: Vec<GAFAnnotation> = Vec::new();
    let mut gene2go: HashMap<String, Vec<String>> = HashMap::default();

    for line in reader.lines().filter_map(Result::ok).filter(|l| !l.starts_with('!')) {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 7 {
            continue;
        }

        let db_object_id = cols[1].to_string();
        let go_term = cols[4].to_string();
        let evidence = cols[6].to_string();
        let gene = cols[2].to_string();

        // Filter out obsolete terms
        if let Some(term) = terms.get(&go_term) {
            if term.is_obsolete {
                continue;
            }
        } else {
            continue;
        }

        // Añadir a la lista de anotaciones
        annotations.push(GAFAnnotation {
            db_object_id: db_object_id.clone(),
            go_term: go_term.clone(),
            evidence,
        });

        // Construir el mapping gene -> GO terms
        gene2go.entry(gene).or_default().push(go_term);
    }

    // Guardar en la caché global
    let _ = GENE2GO_CACHE.set(RwLock::new(gene2go));

    Ok(annotations)
}

/// Build a term counter (counts, IC) from GAF annotations.
///
/// Arguments
/// ---------
/// py_annotations : list of GAFAnnotation
///   List of GAFAnnotation Python objects.
///
/// Returns
/// -------
/// TermCounter
///   Struct with counts and IC values.
#[pyfunction]
pub fn build_term_counter(
    py: Python<'_>,
    py_annotations: Vec<Py<GAFAnnotation>>,
) -> PyResult<TermCounter> {
    // Obtener los términos GO desde el caché global
    let terms = get_terms_or_error()?;

    // Convertir las anotaciones de Py<GAFAnnotation> a GAFAnnotation (Rust)
    let annotations: Vec<GAFAnnotation> = py_annotations
        .into_iter()
        .map(|py_ann| py_ann.extract(py))
        .collect::<PyResult<_>>()?;

    // Llamar a la función de conteo interna
    Ok(_build_term_counter(&annotations, &terms))
}

/// Internal: Build a term counter from Rust GAFAnnotation and GOTerm.
///
/// Arguments
/// ---------
/// annotations : list of GAFAnnotation
///   List of GAFAnnotation structs.
/// terms : dict
///   Map of GO terms.
///
/// Returns
/// -------
/// TermCounter
///   Struct with counts and IC values.
fn _build_term_counter(
    annotations: &[GAFAnnotation],
    terms: &HashMap<String, GOTerm>,
) -> TermCounter {
    use rayon::prelude::*;
    use std::sync::Mutex;

    // Parallel: build obj_to_terms
    let obj_to_terms: HashMap<&str, HashSet<String>> = {
        let obj_to_terms_mutex: Mutex<HashMap<&str, HashSet<String>>> = Mutex::new(HashMap::default());
        annotations.par_iter().for_each(|ann| {
            let go_id = ann.go_term.as_str();
            let mut term_set: HashSet<String> = collect_ancestors(go_id, terms);
            term_set.insert(go_id.to_string());
            let mut map = obj_to_terms_mutex.lock().unwrap();
            map.entry(ann.db_object_id.as_str())
                .or_default()
                .extend(term_set);
        });
        obj_to_terms_mutex.into_inner().unwrap()
    };

    // Parallel: build counts and total_by_ns
    let (counts, total_by_ns) = {
        let counts_mutex: Mutex<HashMap<String, usize>> = Mutex::new(HashMap::default());
        let total_by_ns_mutex: Mutex<HashMap<String, usize>> = Mutex::new(HashMap::default());
        obj_to_terms.values().collect::<Vec<_>>().par_iter().for_each(|term_ids| {
            let mut namespaces_seen = HashSet::default();
            for term_id in *term_ids {
                if let Some(term) = terms.get(term_id.as_str()) {
                    let mut counts = counts_mutex.lock().unwrap();
                    *counts.entry(term_id.to_string()).or_insert(0) += 1;
                    namespaces_seen.insert(term.namespace.as_str());
                }
            }
            for ns in namespaces_seen {
                let mut total_by_ns = total_by_ns_mutex.lock().unwrap();
                *total_by_ns.entry(ns.to_string()).or_insert(0) += 1;
            }
        });
        (counts_mutex.into_inner().unwrap(), total_by_ns_mutex.into_inner().unwrap())
    };

    // Calcular IC (secuencial, as it's fast)
    let mut ic: HashMap<String, f64> = HashMap::default();
    for (term_id, count) in &counts {
        if let Some(term) = terms.get(term_id.as_str()) {
            let total = total_by_ns.get(&term.namespace).copied().unwrap_or(1);
            let freq = *count as f64 / total as f64;
            let info_content = if freq > 0.0 { -freq.ln() } else { 0.0 };
            ic.insert(term_id.clone(), info_content);
        }
    }

    TermCounter {
        counts,
        total_by_ns,
        ic,
    }
}