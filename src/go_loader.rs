use pyo3::prelude::*;
use once_cell::sync::OnceCell;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use std::io::{BufReader, BufRead, Read};
use std::fs::File;
use parking_lot::RwLock;
use std::path::Path;
use std::fs;
use reqwest::blocking::get;
use rayon::prelude::*;
use flate2::read::MultiGzDecoder;

fn open_buf_reader(path: &str) -> std::io::Result<Box<dyn BufRead>> {
    let file = File::open(path)?;
    if path.to_ascii_lowercase().ends_with(".gz") {
        Ok(Box::new(BufReader::new(MultiGzDecoder::new(file))))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

fn read_file_to_string(path: &str) -> std::io::Result<String> {
    let mut reader = open_buf_reader(path)?;
    let mut contents = String::new();
    reader.read_to_string(&mut contents)?;
    Ok(contents)
}

use crate::go_ontology::{GOTerm, PyGOTerm, collect_ancestors, get_terms_or_error};
pub static GO_TERMS_CACHE: OnceCell<RwLock<HashMap<String, GOTerm>>> = OnceCell::new();
pub static OBSOLETE_TERMS_CACHE: OnceCell<RwLock<HashMap<String, GOTerm>>> = OnceCell::new();
pub static GENE2GO_CACHE: OnceCell<RwLock<HashMap<String, Vec<String>>>> = OnceCell::new();
pub static ANCESTORS_CACHE: OnceCell<RwLock<HashMap<String, HashSet<String>>>> = OnceCell::new();
pub static DCA_CACHE: OnceCell<RwLock<HashMap<(String, String), String>>> = OnceCell::new();

fn set_or_replace_cache<T>(cell: &OnceCell<RwLock<T>>, value: T) {
    if let Some(lock) = cell.get() {
        *lock.write() = value;
    } else {
        let _ = cell.set(RwLock::new(value));
    }
}

fn extract_string_set(obj: &Bound<'_, PyAny>) -> PyResult<std::collections::HashSet<String>> {
    let mut set = std::collections::HashSet::new();
    for item in obj.try_iter()? {
        let s: String = item?.extract()?;
        set.insert(s.trim().to_string());
    }
    Ok(set)
}

fn normalize_aspect_token(s: &str) -> String {
    match s.trim().to_ascii_uppercase().as_str() {
        "BP" | "BIOLOGICAL_PROCESS" => "P".to_string(),
        "MF" | "MOLECULAR_FUNCTION" => "F".to_string(),
        "CC" | "CELLULAR_COMPONENT" => "C".to_string(),
        other => other.to_string(),
    }
}

/// Structured filter options applied during GAF/GPAD annotation parsing.
#[pyclass]
#[derive(Clone, Debug)]
pub struct GAFParseOptions {
    pub include_evidence: Option<std::collections::HashSet<String>>,
    pub exclude_evidence: std::collections::HashSet<String>,
    pub include_aspects: Option<std::collections::HashSet<String>>,
    pub exclude_qualifiers: std::collections::HashSet<String>,
    #[pyo3(get, set)]
    pub allow_obsolete_terms: bool,
    #[pyo3(get, set)]
    pub allow_nd_root_only: bool,
}

impl Default for GAFParseOptions {
    fn default() -> Self {
        let mut exclude_qualifiers = std::collections::HashSet::new();
        exclude_qualifiers.insert("NOT".to_string());
        Self {
            include_evidence: None,
            exclude_evidence: std::collections::HashSet::new(),
            include_aspects: None,
            exclude_qualifiers,
            allow_obsolete_terms: false,
            allow_nd_root_only: false,
        }
    }
}

#[pymethods]
impl GAFParseOptions {
    #[new]
    #[pyo3(signature = (
        include_evidence=None,
        exclude_evidence=None,
        include_aspects=None,
        exclude_qualifiers=None,
        allow_obsolete_terms=false,
        allow_nd_root_only=false
    ))]
    pub fn new(
        include_evidence: Option<&Bound<'_, PyAny>>,
        exclude_evidence: Option<&Bound<'_, PyAny>>,
        include_aspects: Option<&Bound<'_, PyAny>>,
        exclude_qualifiers: Option<&Bound<'_, PyAny>>,
        allow_obsolete_terms: bool,
        allow_nd_root_only: bool,
    ) -> PyResult<Self> {
        let inc_ev = match include_evidence {
            Some(obj) => Some(
                extract_string_set(obj)?
                    .into_iter()
                    .map(|s| s.to_ascii_uppercase())
                    .collect(),
            ),
            None => None,
        };
        let exc_ev = match exclude_evidence {
            Some(obj) => extract_string_set(obj)?
                .into_iter()
                .map(|s| s.to_ascii_uppercase())
                .collect(),
            None => std::collections::HashSet::new(),
        };
        let inc_asp = match include_aspects {
            Some(obj) => Some(
                extract_string_set(obj)?
                    .into_iter()
                    .map(|s| normalize_aspect_token(&s))
                    .collect(),
            ),
            None => None,
        };
        let exc_qual = match exclude_qualifiers {
            Some(obj) => extract_string_set(obj)?
                .into_iter()
                .map(|s| s.to_ascii_uppercase())
                .collect(),
            None => {
                let mut def = std::collections::HashSet::new();
                def.insert("NOT".to_string());
                def
            }
        };

        Ok(Self {
            include_evidence: inc_ev,
            exclude_evidence: exc_ev,
            include_aspects: inc_asp,
            exclude_qualifiers: exc_qual,
            allow_obsolete_terms,
            allow_nd_root_only,
        })
    }

    #[getter]
    pub fn include_evidence(&self) -> Option<std::collections::HashSet<String>> {
        self.include_evidence.clone()
    }

    #[setter]
    pub fn set_include_evidence(&mut self, val: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.include_evidence = match val {
            Some(obj) => Some(
                extract_string_set(obj)?
                    .into_iter()
                    .map(|s| s.to_ascii_uppercase())
                    .collect(),
            ),
            None => None,
        };
        Ok(())
    }

    #[getter]
    pub fn exclude_evidence(&self) -> std::collections::HashSet<String> {
        self.exclude_evidence.clone()
    }

    #[setter]
    pub fn set_exclude_evidence(&mut self, val: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.exclude_evidence = match val {
            Some(obj) => extract_string_set(obj)?
                .into_iter()
                .map(|s| s.to_ascii_uppercase())
                .collect(),
            None => std::collections::HashSet::new(),
        };
        Ok(())
    }

    #[getter]
    pub fn include_aspects(&self) -> Option<std::collections::HashSet<String>> {
        self.include_aspects.clone()
    }

    #[setter]
    pub fn set_include_aspects(&mut self, val: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.include_aspects = match val {
            Some(obj) => Some(
                extract_string_set(obj)?
                    .into_iter()
                    .map(|s| normalize_aspect_token(&s))
                    .collect(),
            ),
            None => None,
        };
        Ok(())
    }

    #[getter]
    pub fn exclude_qualifiers(&self) -> std::collections::HashSet<String> {
        self.exclude_qualifiers.clone()
    }

    #[setter]
    pub fn set_exclude_qualifiers(&mut self, val: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.exclude_qualifiers = match val {
            Some(obj) => extract_string_set(obj)?
                .into_iter()
                .map(|s| s.to_ascii_uppercase())
                .collect(),
            None => std::collections::HashSet::new(),
        };
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "GAFParseOptions(include_evidence={:?}, exclude_evidence={:?}, include_aspects={:?}, exclude_qualifiers={:?}, allow_obsolete_terms={}, allow_nd_root_only={})",
            self.include_evidence,
            self.exclude_evidence,
            self.include_aspects,
            self.exclude_qualifiers,
            self.allow_obsolete_terms,
            self.allow_nd_root_only
        )
    }
}

/// Struct representing a single annotation from a GAF 2.2 file.
///
/// Fields
/// ------
/// db : str
///   Database source (e.g., UniProtKB).
/// db_object_id : str
///   The gene product identifier (e.g., UniProt ID).
/// db_object_symbol : str
///   Gene product symbol (e.g., TP53).
/// qualifiers : list of str
///   Annotation qualifiers (e.g., ["enables"] or ["NOT", "involved_in"]).
/// go_id : str
///   The GO term ID (e.g., GO:0008150). Also accessible via `.go_term`.
/// references : list of str
///   Database references supporting the annotation (e.g., ["PMID:21873635"]).
/// evidence : str
///   The evidence code for the annotation (e.g., IEA, IDA).
/// with_from : list of str
///   With/From field values.
/// aspect : str
///   Ontology aspect ("P", "F", or "C").
/// db_object_name : str
///   Full name of the gene product.
/// db_object_synonyms : list of str
///   Synonyms for the gene product.
/// db_object_type : str
///   Type of gene product (e.g., protein).
/// taxon : list of str
///   Taxon identifiers (e.g., ["taxon:9606"]).
/// date : str
///   Date of annotation (YYYYMMDD).
/// assigned_by : str
///   Database which made the annotation.
/// annotation_extensions : list of str
///   Cross-links to other ontologies/terms.
/// gene_product_form_id : str
///   Specific isoform or post-translationally modified form ID.
/// is_negated : bool
///   True if the annotation is negated ("NOT" in qualifiers).
#[pyclass]
#[derive(Clone, Debug)]
pub struct GAFAnnotation {
    #[pyo3(get)]
    pub db: String,
    #[pyo3(get)]
    pub db_object_id: String,
    #[pyo3(get)]
    pub db_object_symbol: String,
    #[pyo3(get)]
    pub qualifiers: Vec<String>,
    #[pyo3(get)]
    pub go_id: String,
    #[pyo3(get)]
    pub references: Vec<String>,
    #[pyo3(get)]
    pub evidence: String,
    #[pyo3(get)]
    pub with_from: Vec<String>,
    #[pyo3(get)]
    pub aspect: String,
    #[pyo3(get)]
    pub db_object_name: String,
    #[pyo3(get)]
    pub db_object_synonyms: Vec<String>,
    #[pyo3(get)]
    pub db_object_type: String,
    #[pyo3(get)]
    pub taxon: Vec<String>,
    #[pyo3(get)]
    pub date: String,
    #[pyo3(get)]
    pub assigned_by: String,
    #[pyo3(get)]
    pub annotation_extensions: Vec<String>,
    #[pyo3(get)]
    pub gene_product_form_id: String,
    #[pyo3(get)]
    pub is_negated: bool,
}

#[pymethods]
impl GAFAnnotation {
    #[new]
    #[pyo3(signature = (
        db_object_id,
        go_id,
        evidence,
        db="".to_string(),
        db_object_symbol="".to_string(),
        qualifiers=Vec::new(),
        references=Vec::new(),
        with_from=Vec::new(),
        aspect="".to_string(),
        db_object_name="".to_string(),
        db_object_synonyms=Vec::new(),
        db_object_type="".to_string(),
        taxon=Vec::new(),
        date="".to_string(),
        assigned_by="".to_string(),
        annotation_extensions=Vec::new(),
        gene_product_form_id="".to_string(),
        is_negated=None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        db_object_id: String,
        go_id: String,
        evidence: String,
        db: String,
        db_object_symbol: String,
        qualifiers: Vec<String>,
        references: Vec<String>,
        with_from: Vec<String>,
        aspect: String,
        db_object_name: String,
        db_object_synonyms: Vec<String>,
        db_object_type: String,
        taxon: Vec<String>,
        date: String,
        assigned_by: String,
        annotation_extensions: Vec<String>,
        gene_product_form_id: String,
        is_negated: Option<bool>,
    ) -> Self {
        let negated = is_negated.unwrap_or_else(|| {
            qualifiers
                .iter()
                .any(|q| q.eq_ignore_ascii_case("NOT") || q.to_ascii_uppercase().starts_with("NOT|"))
        });
        Self {
            db,
            db_object_id,
            db_object_symbol,
            qualifiers,
            go_id,
            references,
            evidence,
            with_from,
            aspect,
            db_object_name,
            db_object_synonyms,
            db_object_type,
            taxon,
            date,
            assigned_by,
            annotation_extensions,
            gene_product_form_id,
            is_negated: negated,
        }
    }

    /// Alias for `go_id` preserved for backwards compatibility.
    #[getter]
    pub fn go_term(&self) -> &str {
        &self.go_id
    }

    fn __repr__(&self) -> String {
        format!(
            "GAFAnnotation(db='{}', db_object_id='{}', db_object_symbol='{}', qualifiers={:?}, go_id='{}', evidence='{}', aspect='{}', is_negated={})",
            self.db,
            self.db_object_id,
            self.db_object_symbol,
            self.qualifiers,
            self.go_id,
            self.evidence,
            self.aspect,
            self.is_negated
        )
    }
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
    let contents = read_file_to_string(path).expect("Can't open OBO file");
    let chunks = contents.split("[Term]");

    let all_parsed_terms: Vec<GOTerm> = chunks
        .par_bridge()
        .filter_map(parse_term_chunk)
        .collect();

    let mut canonical_terms = Vec::with_capacity(all_parsed_terms.len());
    let mut obsolete_map: HashMap<String, GOTerm> = HashMap::default();

    for term in all_parsed_terms {
        if term.is_obsolete {
            let mut all_ids = term.alt_ids.clone();
            all_ids.push(term.id.clone());
            for id in &all_ids {
                let mut clone = term.clone();
                clone.id = id.clone();
                clone.alt_ids = all_ids.clone();
                obsolete_map.insert(id.clone(), clone);
            }
        } else {
            canonical_terms.push(term);
        }
    }
    set_or_replace_cache(&OBSOLETE_TERMS_CACHE, obsolete_map);

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

    fn collect_ancestors_uncached(go_id: &str, terms: &HashMap<String, GOTerm>) -> HashSet<String> {
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
        visited
    }

    // Precompute ancestors for caching
    let ancestors_map: HashMap<String, HashSet<String>> = term_map
        .par_iter()
        .map(|(id, _)| {
            let ancestors = collect_ancestors_uncached(id, &term_map);
            (id.clone(), ancestors)
        })
        .collect();
    set_or_replace_cache(&ANCESTORS_CACHE, ancestors_map);

    // Initialize (or reset) DCA_CACHE
    set_or_replace_cache(&DCA_CACHE, HashMap::default());

    // Clear caches in semantic similarity code that depend on the GO graph topology.
    crate::go_semantic::clear_internal_caches();

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

    let url = "https://purl.obolibrary.org/obo/go/go-basic.obo";
    println!("Descargando ontología desde: {}", url);
    let response = get(url)
        .and_then(|r| r.error_for_status())
        .map_err(|e| e.to_string())?;

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
    let path = match path {
        Some(p) => p,
        None => download_obo()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?,
    };
    if !Path::new(&path).exists() {
        return Err(pyo3::exceptions::PyIOError::new_err(format!(
            "OBO file not found: {}",
            path
        )));
    }
    let terms_map = parse_obo(&path);

    // Guardar en la caché global
    set_or_replace_cache(&GO_TERMS_CACHE, terms_map.clone());

    // Devolver lista de PyGOTerm
    let terms_vec = terms_map
        .into_iter()
        .map(|(_, v)| PyGOTerm::from(&v))
        .collect();

    Ok(terms_vec)
}

#[inline]
fn split_pipe(s: &str) -> Vec<String> {
    let s = s.trim();
    if s.is_empty() {
        Vec::new()
    } else {
        s.split('|')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(String::from)
            .collect()
    }
}

#[inline]
pub(crate) fn is_root_term(go_id: &str, terms: &HashMap<String, GOTerm>) -> bool {
    matches!(go_id, "GO:0008150" | "GO:0003674" | "GO:0005575")
        || terms.get(go_id).map_or(false, |t| t.parents.is_empty())
}

#[inline]
pub(crate) fn namespace_to_aspect(ns: &str) -> &'static str {
    match ns {
        "biological_process" => "P",
        "molecular_function" => "F",
        "cellular_component" => "C",
        _ => "",
    }
}

/// Apply `GAFParseOptions` filters and obsolete-term resolution to a candidate annotation.
/// Returns `Some(resolved_go_id)` if the annotation passes all filters, or `None` if filtered out.
pub(crate) fn filter_and_resolve_annotation(
    raw_go_id: &str,
    raw_qualifiers: &[String],
    evidence: &str,
    aspect: &str,
    opts: &GAFParseOptions,
    terms: &HashMap<String, GOTerm>,
    obsolete_terms: Option<&HashMap<String, GOTerm>>,
) -> Option<String> {
    // 1. Resolve GO term and check obsolete status
    let go_id = if let Some(term) = terms.get(raw_go_id) {
        if term.is_obsolete {
            if !opts.allow_obsolete_terms {
                return None;
            }
            if let Some(ref replacement) = term.replaced_by {
                if terms.contains_key(replacement) {
                    replacement.clone()
                } else {
                    raw_go_id.to_string()
                }
            } else if let Some(first_valid) = term.consider.iter().find(|c| terms.contains_key(*c)) {
                first_valid.clone()
            } else {
                raw_go_id.to_string()
            }
        } else {
            raw_go_id.to_string()
        }
    } else if let Some(obs_term) = obsolete_terms.and_then(|m| m.get(raw_go_id)) {
        if !opts.allow_obsolete_terms {
            return None;
        }
        if let Some(ref replacement) = obs_term.replaced_by {
            if terms.contains_key(replacement) {
                replacement.clone()
            } else {
                raw_go_id.to_string()
            }
        } else if let Some(first_valid) = obs_term.consider.iter().find(|c| terms.contains_key(*c)) {
            first_valid.clone()
        } else {
            raw_go_id.to_string()
        }
    } else {
        return None;
    };

    // 2. Check aspect filter
    if let Some(ref inc_aspects) = opts.include_aspects {
        let effective_aspect = if !aspect.is_empty() {
            normalize_aspect_token(aspect)
        } else if let Some(t) = terms.get(&go_id) {
            namespace_to_aspect(&t.namespace).to_string()
        } else if let Some(t) = obsolete_terms.and_then(|m| m.get(&go_id)) {
            namespace_to_aspect(&t.namespace).to_string()
        } else {
            String::new()
        };
        if !inc_aspects.contains(&effective_aspect) {
            return None;
        }
    }

    // 3. Check qualifier filter
    if !opts.exclude_qualifiers.is_empty() {
        for q in raw_qualifiers {
            if opts.exclude_qualifiers.contains(&q.to_ascii_uppercase()) {
                return None;
            }
        }
    }

    // 4. Check evidence filters
    let ev_upper = evidence.trim().to_ascii_uppercase();
    if opts.exclude_evidence.contains(&ev_upper) {
        return None;
    }
    if let Some(ref inc_ev) = opts.include_evidence {
        if !inc_ev.contains(&ev_upper) {
            return None;
        }
    }

    // 5. Handle ND (No biological Data)
    if ev_upper == "ND" {
        if opts.allow_nd_root_only {
            if !is_root_term(&go_id, terms) {
                return None;
            }
        } else {
            return None;
        }
    }

    Some(go_id)
}

/// Load a GAF annotation file and cache the gene-to-GO mapping.
///
/// Arguments
/// ---------
/// path : str
///   Path to the GAF file.
/// options : Optional[GAFParseOptions]
///   Optional structured filter options applied during parsing.
///
/// Returns
/// -------
/// list of GAFAnnotation
///   List of parsed GAF annotations.
#[pyfunction]
#[pyo3(signature = (path, options=None))]
pub fn load_gaf(path: String, options: Option<GAFParseOptions>) -> PyResult<Vec<GAFAnnotation>> {
    let opts = options.unwrap_or_default();
    let reader = open_buf_reader(&path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    // Get the loaded GO terms to check for obsolete terms
    let terms = crate::go_ontology::get_terms_or_error()?;
    let obsolete_guard = OBSOLETE_TERMS_CACHE.get().map(|lock| lock.read());
    let obsolete_terms = obsolete_guard.as_deref();

    let mut annotations: Vec<GAFAnnotation> = Vec::new();
    let mut gene2go: HashMap<String, Vec<String>> = HashMap::default();

    for line in reader.lines().filter_map(Result::ok).filter(|l| !l.starts_with('!')) {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 7 {
            continue;
        }

        let raw_go_id = cols[4].trim();
        let evidence_str = cols[6].trim();
        let aspect_str = cols.get(8).copied().unwrap_or("").trim();
        let qualifiers = split_pipe(cols[3]);

        let Some(go_id) = filter_and_resolve_annotation(
            raw_go_id,
            &qualifiers,
            evidence_str,
            aspect_str,
            &opts,
            &terms,
            obsolete_terms,
        ) else {
            continue;
        };

        let db = cols[0].trim().to_string();
        let db_object_id = cols[1].trim().to_string();
        let db_object_symbol = cols[2].trim().to_string();
        let references = cols.get(5).map(|s| split_pipe(s)).unwrap_or_default();
        let evidence = evidence_str.to_string();
        let with_from = cols.get(7).map(|s| split_pipe(s)).unwrap_or_default();
        let aspect = if !aspect_str.is_empty() {
            aspect_str.to_string()
        } else if let Some(t) = terms.get(&go_id) {
            namespace_to_aspect(&t.namespace).to_string()
        } else {
            String::new()
        };
        let db_object_name = cols.get(9).copied().unwrap_or("").trim().to_string();
        let db_object_synonyms = cols.get(10).map(|s| split_pipe(s)).unwrap_or_default();
        let db_object_type = cols.get(11).copied().unwrap_or("").trim().to_string();
        let taxon = cols.get(12).map(|s| split_pipe(s)).unwrap_or_default();
        let date = cols.get(13).copied().unwrap_or("").trim().to_string();
        let assigned_by = cols.get(14).copied().unwrap_or("").trim().to_string();
        let annotation_extensions = cols.get(15).map(|s| split_pipe(s)).unwrap_or_default();
        let gene_product_form_id = cols.get(16).copied().unwrap_or("").trim().to_string();
        let is_negated = qualifiers
            .iter()
            .any(|q| q.eq_ignore_ascii_case("NOT"));

        let gene_key = if !db_object_symbol.is_empty() {
            db_object_symbol.clone()
        } else {
            db_object_id.clone()
        };

        annotations.push(GAFAnnotation {
            db,
            db_object_id,
            db_object_symbol,
            qualifiers,
            go_id: go_id.clone(),
            references,
            evidence,
            with_from,
            aspect,
            db_object_name,
            db_object_synonyms,
            db_object_type,
            taxon,
            date,
            assigned_by,
            annotation_extensions,
            gene_product_form_id,
            is_negated,
        });

        gene2go.entry(gene_key).or_default().push(go_id);
    }

    // Save in global cache
    set_or_replace_cache(&GENE2GO_CACHE, gene2go);

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

    // Extraer únicamente (db_object_id, go_id) sin clonar todos los vectores de cada GAFAnnotation
    let pairs: Vec<(String, String)> = py_annotations
        .iter()
        .map(|py_ann| {
            let b = py_ann.borrow(py);
            (b.db_object_id.clone(), b.go_id.clone())
        })
        .collect();

    Ok(_build_term_counter_pairs(&pairs, &terms))
}

/// Internal: Build a term counter from (db_object_id, go_id) pairs and GOTerm map.
fn _build_term_counter_pairs(
    annotations: &[(String, String)],
    terms: &HashMap<String, GOTerm>,
) -> TermCounter {
    let ancestors_cache_guard = ANCESTORS_CACHE.get().map(|lock| lock.read());
    let ancestors_cache = ancestors_cache_guard.as_deref();
    let obsolete_guard = OBSOLETE_TERMS_CACHE.get().map(|lock| lock.read());
    let obsolete_terms = obsolete_guard.as_deref();

    // Parallel: build obj_to_terms without global locking
    let obj_to_terms: HashMap<&str, HashSet<String>> = annotations
        .par_iter()
        .fold(HashMap::<&str, HashSet<String>>::default, |mut acc, (db_obj_id, go_id_str)| {
            let go_id = go_id_str.as_str();
            let entry = acc.entry(db_obj_id.as_str()).or_default();

            entry.insert(go_id.to_string());
            if let Some(cache) = ancestors_cache {
                if let Some(ancestors) = cache.get(go_id) {
                    entry.extend(ancestors.iter().cloned());
                } else {
                    entry.extend(collect_ancestors(go_id, terms).into_iter());
                }
            } else {
                entry.extend(collect_ancestors(go_id, terms).into_iter());
            }
            acc
        })
        .reduce(HashMap::<&str, HashSet<String>>::default, |mut acc, map| {
            for (k, v) in map {
                acc.entry(k).or_default().extend(v);
            }
            acc
        });

    // Parallel: build counts and total_by_ns without per-term locks
    let (counts, total_by_ns) = obj_to_terms
        .par_iter()
        .fold(
            || (
                HashMap::<String, usize>::default(),
                HashMap::<String, usize>::default(),
            ),
            |(mut counts, mut total_by_ns), (_gene, term_ids)| {
                let mut namespaces_seen: HashSet<&str> = HashSet::default();
                for term_id in term_ids {
                    if let Some(term) = terms.get(term_id.as_str()) {
                        *counts.entry(term_id.clone()).or_insert(0) += 1;
                        namespaces_seen.insert(term.namespace.as_str());
                    } else if let Some(obs_term) = obsolete_terms.and_then(|m| m.get(term_id.as_str())) {
                        *counts.entry(term_id.clone()).or_insert(0) += 1;
                        namespaces_seen.insert(obs_term.namespace.as_str());
                    }
                }
                for ns in namespaces_seen {
                    *total_by_ns.entry(ns.to_string()).or_insert(0) += 1;
                }
                (counts, total_by_ns)
            },
        )
        .reduce(
            || (
                HashMap::<String, usize>::default(),
                HashMap::<String, usize>::default(),
            ),
            |(mut counts_a, mut total_a), (counts_b, total_b)| {
                for (k, v) in counts_b {
                    *counts_a.entry(k).or_insert(0) += v;
                }
                for (k, v) in total_b {
                    *total_a.entry(k).or_insert(0) += v;
                }
                (counts_a, total_a)
            },
        );

    // Calcular IC
    let mut ic: HashMap<String, f64> = HashMap::default();
    for (term_id, count) in &counts {
        let ns_opt = terms
            .get(term_id.as_str())
            .map(|t| t.namespace.as_str())
            .or_else(|| {
                obsolete_terms
                    .and_then(|m| m.get(term_id.as_str()))
                    .map(|t| t.namespace.as_str())
            });
        if let Some(ns) = ns_opt {
            let total = total_by_ns.get(ns).copied().unwrap_or(1);
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
