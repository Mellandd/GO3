use pyo3::prelude::*;
use once_cell::sync::OnceCell;
use std::collections::{HashMap, HashSet};
use std::io::{BufReader, BufRead};
use std::fs::File;
use std::sync::RwLock;
use std::path::Path;
use std::fs;
use reqwest::blocking::get;
use rayon::prelude::*;

use crate::go_ontology::{GOTerm, PyGOTerm, collect_ancestors, get_terms_or_error};
pub static GO_TERMS_CACHE: OnceCell<RwLock<HashMap<String, GOTerm>>> = OnceCell::new();

#[pyclass]
#[derive(Clone)]
pub struct GAFAnnotation {
    #[pyo3(get)]
    pub db_object_id: String,  // por ejemplo P12345
    #[pyo3(get)]
    pub go_term: String,       // por ejemplo GO:0008150
    #[pyo3(get)]
    pub evidence: String,      // por ejemplo IEA
}

#[pyclass]
#[derive(Clone)]
pub struct TermCounter {
    #[pyo3(get)]
    pub counts: HashMap<String, usize>,
    #[pyo3(get)]
    pub total: usize,
    #[pyo3(get)]
    pub ic: HashMap<String, f64>,
}

pub fn parse_obo(path: &str) -> HashMap<String, GOTerm> {
    let contents = fs::read_to_string(path).expect("Can't open OBO file");
    let chunks = contents.split("[Term]");

    let terms: Vec<GOTerm> = chunks
        .par_bridge() // menos overhead si no es necesario `.par_iter()`
        .filter_map(parse_term_chunk)
        .collect();

    let mut term_map = HashMap::with_capacity(terms.len());
    for term in terms {
        term_map.insert(term.id.clone(), term);
    }

    compute_levels_and_depths(&mut term_map);
    term_map
}

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

    let lines = chunk.lines().map(str::trim);
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
            let parent = line["is_a: ".len()..].split_whitespace().next().unwrap_or("").to_string();
            term.parents.push(parent);
        } else if line.starts_with("alt_id: ") {
            term.alt_ids.push(line["alt_id: ".len()..].to_string());
        } else if line.starts_with("replaced_by: ") {
            term.replaced_by = Some(line["replaced_by: ".len()..].to_string());
        } else if line.starts_with("consider: ") {
            term.consider.push(line["consider: ".len()..].to_string());
        } else if line.starts_with("synonym: ") {
            term.synonyms.push(line["synonym: ".len()..].to_string());
        } else if line.starts_with("xref: ") {
            term.xrefs.push(line["xref: ".len()..].to_string());
        } else if line.starts_with("relationship: ") {
            let mut parts = line["relationship: ".len()..].split_whitespace();
            if let (Some(rel_type), Some(target)) = (parts.next(), parts.next()) {
                term.relationships.push((rel_type.to_string(), target.to_string()));
            }
        } else if line.starts_with("comment: ") {
            term.comment = Some(line["comment: ".len()..].to_string());
        } else if line.starts_with("is_obsolete: true") {
            term.is_obsolete = true;
        }
    }

    if valid && !term.id.is_empty() && term.id.starts_with("GO:") {
        Some(term)
    } else {
        None
    }
}

pub fn compute_levels_and_depths(terms: &mut HashMap<String, GOTerm>) {
    // Paso 1: construir mapa de hijos
    let mut child_map: HashMap<String, Vec<String>> = HashMap::new();
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
            eprintln!("⚠️ Ciclo detectado en depth: {}", term_id);
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
        let mut visiting = HashSet::new();
        init_level(id, terms, &mut visiting);

        let mut visiting = HashSet::new();
        init_depth(id, terms, &mut visiting);
    }

    // Paso 5: rellenar el campo children con los hijos (solo vía is_a)
    for (parent, children) in child_map {
        if let Some(term) = terms.get_mut(&parent) {
            term.children = children;
        }
    }
}

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

#[pyfunction]
pub fn load_gaf(path: String) -> PyResult<Vec<GAFAnnotation>> {
    let file = File::open(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let reader = BufReader::new(file);

    let annotations = reader
        .lines()
        .filter_map(Result::ok)
        .filter(|line| !line.starts_with('!'))
        .filter_map(|line| {
            let cols: Vec<&str> = line.split('\t').collect();
            if cols.len() < 7 {
                return None;
            }
            Some(GAFAnnotation {
                db_object_id: cols[1].to_string(),
                go_term: cols[4].to_string(),
                evidence: cols[6].to_string(),
            })
        })
        .collect();

    Ok(annotations)
}

#[pyfunction]
pub fn build_term_counter(annotations: Vec<GAFAnnotation>) -> PyResult<TermCounter> {
    // Load GO terms with proper error handling
    let terms = get_terms_or_error()?;
    
    // Pre-compute ancestor cache for all terms
    let ancestor_cache: HashMap<&str, HashSet<&str>> = terms
        .keys()
        .map(|id| id.as_str())
        .collect::<Vec<_>>()
        .par_iter()
        .map(|&id| (id, collect_ancestors(id, &terms)))
        .collect();

    // Phase 1: Group annotations by gene in parallel
    let genes_to_terms: HashMap<String, HashSet<&str>> = annotations
        .par_iter()
        .fold(
            || HashMap::new(),
            |mut acc: HashMap<String, HashSet<&str>>, annotation| {
                acc.entry(annotation.db_object_id.clone())
                   .or_default()
                   .insert(annotation.go_term.as_str());
                acc
            }
        )
        .reduce(
            || HashMap::new(),
            |mut a, b| {
                for (gene, terms) in b {
                    a.entry(gene).or_default().extend(terms);
                }
                a
            }
        );

    let total_genes = genes_to_terms.len();

    // Phase 2: Count term occurrences in parallel
    let counts: HashMap<String, usize> = genes_to_terms
        .par_iter()
        .fold(
            || HashMap::new(),
            |mut acc: HashMap<String, usize>, (_, gene_terms)| {
                let mut all_ancestors = HashSet::new();
                for &term in gene_terms {
                    if let Some(ancestors) = ancestor_cache.get(term) {
                        all_ancestors.extend(ancestors.iter().copied());
                    }
                }
                for &ancestor in &all_ancestors {
                    *acc.entry(ancestor.to_string()).or_default() += 1;
                }
                acc
            }
        )
        .reduce(
            || HashMap::new(),
            |mut a, b| {
                for (term, count) in b {
                    *a.entry(term).or_default() += count;
                }
                a
            }
        );

    // Phase 3: Calculate IC in parallel
    let ic: HashMap<String, f64> = counts
        .par_iter()
        .map(|(term, count)| {
            let probability = (*count as f64) / (total_genes as f64);
            let information_content = -probability.log2();
            (term.clone(), information_content)
        })
        .collect();

    Ok(TermCounter {
        counts,
        total: total_genes,
        ic,
    })
}