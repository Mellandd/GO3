use pyo3::exceptions::{PyImportError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::go_loader::TermCounter;

use super::gene::gene_distance_matrix;

/// Compute t-SNE embeddings from a gene list using a precomputed distance matrix.
#[pyfunction]
#[pyo3(signature = (genes=None, ontology="BP", similarity="lin", groupwise="bma", counter=None, distance_transform="auto", n_components=2, perplexity=30.0, n_iter=1000, random_state=None))]
pub fn tsne_genes(
    py: Python<'_>,
    genes: Option<Vec<String>>,
    ontology: &str,
    similarity: &str,
    groupwise: &str,
    counter: Option<&TermCounter>,
    distance_transform: &str,
    n_components: usize,
    perplexity: f64,
    n_iter: usize,
    random_state: Option<u64>,
) -> PyResult<(Vec<String>, Py<PyAny>)> {
    let (gene_list, matrix) = gene_distance_matrix(
        genes,
        ontology,
        similarity,
        groupwise,
        counter,
        distance_transform,
    )?;
    let n = gene_list.len();
    if n < 2 {
        return Err(PyValueError::new_err(
            "At least two genes are required for t-SNE",
        ));
    }
    if perplexity >= n as f64 {
        return Err(PyValueError::new_err(format!(
            "perplexity ({}) must be less than number of genes ({})",
            perplexity, n
        )));
    }

    let sklearn = py.import("sklearn.manifold").map_err(|_| {
        PyImportError::new_err("scikit-learn is required for tsne_genes. Install with `pip install scikit-learn`.")
    })?;
    let tsne_class = sklearn.getattr("TSNE")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("metric", "precomputed")?;
    kwargs.set_item("n_components", n_components)?;
    kwargs.set_item("perplexity", perplexity)?;
    kwargs.set_item("init", "random")?;
    // scikit-learn >= 1.4 uses `max_iter`, older versions use `n_iter`
    let inspect = py.import("inspect")?;
    let sig = inspect.call_method1("signature", (tsne_class.clone(),))?;
    let params = sig.getattr("parameters")?;
    let has_max_iter = params
        .call_method1("__contains__", ("max_iter",))?
        .is_truthy()?;
    if has_max_iter {
        kwargs.set_item("max_iter", n_iter)?;
    } else {
        kwargs.set_item("n_iter", n_iter)?;
    }
    if let Some(rs) = random_state {
        kwargs.set_item("random_state", rs)?;
        // Avoid UMAP warning about n_jobs being overridden when a seed is set.
        kwargs.set_item("n_jobs", 1)?;
    }
    let tsne = tsne_class.call((), Some(&kwargs))?;
    let dist_py = PyList::new(py, matrix)?;
    let numpy = py.import("numpy")?;
    let dist_np = numpy.call_method1("asarray", (dist_py,))?;
    let embedding = tsne.call_method1("fit_transform", (dist_np,))?;

    Ok((gene_list, embedding.into()))
}

/// Compute UMAP embeddings from a gene list using a precomputed distance matrix.
#[pyfunction]
#[pyo3(signature = (genes=None, ontology="BP", similarity="lin", groupwise="bma", counter=None, distance_transform="auto", n_components=2, n_neighbors=15, min_dist=0.1, random_state=None))]
pub fn umap_genes(
    py: Python<'_>,
    genes: Option<Vec<String>>,
    ontology: &str,
    similarity: &str,
    groupwise: &str,
    counter: Option<&TermCounter>,
    distance_transform: &str,
    n_components: usize,
    n_neighbors: usize,
    min_dist: f64,
    random_state: Option<u64>,
) -> PyResult<(Vec<String>, Py<PyAny>)> {
    let (gene_list, matrix) = gene_distance_matrix(
        genes,
        ontology,
        similarity,
        groupwise,
        counter,
        distance_transform,
    )?;
    let n = gene_list.len();
    if n < 2 {
        return Err(PyValueError::new_err(
            "At least two genes are required for UMAP",
        ));
    }
    if n_neighbors >= n {
        return Err(PyValueError::new_err(format!(
            "n_neighbors ({}) must be less than number of genes ({})",
            n_neighbors, n
        )));
    }

    let umap_mod = py.import("umap").map_err(|_| {
        PyImportError::new_err("umap-learn is required for umap_genes. Install with `pip install umap-learn`.")
    })?;
    let umap_class = umap_mod.getattr("UMAP")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("metric", "precomputed")?;
    kwargs.set_item("n_components", n_components)?;
    kwargs.set_item("n_neighbors", n_neighbors)?;
    kwargs.set_item("min_dist", min_dist)?;
    if let Some(rs) = random_state {
        kwargs.set_item("random_state", rs)?;
    }
    let model = umap_class.call((), Some(&kwargs))?;
    let dist_py = PyList::new(py, matrix)?;
    let numpy = py.import("numpy")?;
    let dist_np = numpy.call_method1("asarray", (dist_py,))?;
    let embedding = model.call_method1("fit_transform", (dist_np,))?;

    Ok((gene_list, embedding.into()))
}

/// Plot a 2D embedding with matplotlib.
#[pyfunction]
#[pyo3(signature = (embedding, genes=None, labels=None, title=None, annotate="auto", max_labels=200, figsize=(6.0, 5.0), s=18.0, alpha=0.85, ax=None))]
pub fn plot_embedding(
    py: Python<'_>,
    embedding: Py<PyAny>,
    genes: Option<Vec<String>>,
    labels: Option<Vec<String>>,
    title: Option<String>,
    annotate: &str,
    max_labels: usize,
    figsize: (f64, f64),
    s: f64,
    alpha: f64,
    ax: Option<Py<PyAny>>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let _plt = py.import("matplotlib.pyplot").map_err(|_| {
        PyImportError::new_err("matplotlib is required for plot_embedding. Install with `pip install go3[viz]`.")
    })?;

    let locals = PyDict::new(py);
    locals.set_item("embedding", embedding)?;
    match genes {
        Some(list) => locals.set_item("genes", list)?,
        None => locals.set_item("genes", py.None())?,
    };
    match labels {
        Some(list) => locals.set_item("labels", list)?,
        None => locals.set_item("labels", py.None())?,
    };
    match title {
        Some(text) => locals.set_item("title", text)?,
        None => locals.set_item("title", py.None())?,
    };
    locals.set_item("annotate", annotate)?;
    locals.set_item("max_labels", max_labels)?;
    locals.set_item("figsize", (figsize.0, figsize.1))?;
    locals.set_item("s", s)?;
    locals.set_item("alpha", alpha)?;
    match ax {
        Some(ax_obj) => locals.set_item("ax", ax_obj)?,
        None => locals.set_item("ax", py.None())?,
    };

    py.run(
        pyo3::ffi::c_str!(
            r#"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

emb = np.asarray(embedding)
if emb.ndim != 2 or emb.shape[1] < 2:
    raise ValueError("embedding must be 2D with at least 2 columns")

x = emb[:, 0]
y = emb[:, 1]

if ax is None:
    fig, ax = plt.subplots(figsize=figsize)
else:
    fig = ax.figure

if labels is None:
    ax.scatter(x, y, s=s, alpha=alpha)
else:
    if len(labels) != emb.shape[0]:
        raise ValueError("labels length must match embedding")
    lab_arr = np.asarray(labels)
    unique_labels = []
    seen = set()
    for lab in lab_arr:
        key = str(lab)
        if key not in seen:
            unique_labels.append(lab)
            seen.add(key)
    cmap = plt.get_cmap("tab20", len(unique_labels))
    for idx, lab in enumerate(unique_labels):
        mask = lab_arr == lab
        ax.scatter(x[mask], y[mask], s=s, alpha=alpha, color=cmap(idx), label=str(lab))
    ax.legend(loc="best", fontsize=8, frameon=False)

if title is not None:
    ax.set_title(title)

ann = (annotate or "auto").lower()
if genes is not None and ann != "none":
    if len(genes) != emb.shape[0]:
        raise ValueError("genes length must match embedding")
    do_annotate = (ann == "all") or (ann == "auto" and len(genes) <= max_labels)
    if do_annotate:
        for xi, yi, g in zip(x, y, genes):
            ax.text(xi, yi, g, fontsize=8)

ax.set_xlabel("dim 1")
ax.set_ylabel("dim 2")
"#
        ),
        None,
        Some(&locals),
    )?;

    let fig = locals
        .get_item("fig")?
        .ok_or_else(|| PyValueError::new_err("Plotting failed to create figure"))?
        .unbind();
    let ax = locals
        .get_item("ax")?
        .ok_or_else(|| PyValueError::new_err("Plotting failed to create axes"))?
        .unbind();

    Ok((fig, ax))
}

/// Compute t-SNE embeddings and plot them with matplotlib.
#[pyfunction]
#[pyo3(signature = (genes=None, ontology="BP", similarity="lin", groupwise="bma", counter=None, distance_transform="auto", n_components=2, perplexity=30.0, n_iter=1000, random_state=None, labels=None, title=None, annotate="auto", max_labels=200, figsize=(6.0, 5.0), s=18.0, alpha=0.85, ax=None))]
pub fn plot_tsne_genes(
    py: Python<'_>,
    genes: Option<Vec<String>>,
    ontology: &str,
    similarity: &str,
    groupwise: &str,
    counter: Option<&TermCounter>,
    distance_transform: &str,
    n_components: usize,
    perplexity: f64,
    n_iter: usize,
    random_state: Option<u64>,
    labels: Option<Vec<String>>,
    title: Option<String>,
    annotate: &str,
    max_labels: usize,
    figsize: (f64, f64),
    s: f64,
    alpha: f64,
    ax: Option<Py<PyAny>>,
) -> PyResult<(Vec<String>, Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let (gene_list, embedding) = tsne_genes(
        py,
        genes,
        ontology,
        similarity,
        groupwise,
        counter,
        distance_transform,
        n_components,
        perplexity,
        n_iter,
        random_state,
    )?;

    let (fig, ax_obj) = plot_embedding(
        py,
        embedding.clone_ref(py),
        Some(gene_list.clone()),
        labels,
        title,
        annotate,
        max_labels,
        figsize,
        s,
        alpha,
        ax,
    )?;

    Ok((gene_list, embedding, fig, ax_obj))
}

/// Compute UMAP embeddings and plot them with matplotlib.
#[pyfunction]
#[pyo3(signature = (genes=None, ontology="BP", similarity="lin", groupwise="bma", counter=None, distance_transform="auto", n_components=2, n_neighbors=15, min_dist=0.1, random_state=None, labels=None, title=None, annotate="auto", max_labels=200, figsize=(6.0, 5.0), s=18.0, alpha=0.85, ax=None))]
pub fn plot_umap_genes(
    py: Python<'_>,
    genes: Option<Vec<String>>,
    ontology: &str,
    similarity: &str,
    groupwise: &str,
    counter: Option<&TermCounter>,
    distance_transform: &str,
    n_components: usize,
    n_neighbors: usize,
    min_dist: f64,
    random_state: Option<u64>,
    labels: Option<Vec<String>>,
    title: Option<String>,
    annotate: &str,
    max_labels: usize,
    figsize: (f64, f64),
    s: f64,
    alpha: f64,
    ax: Option<Py<PyAny>>,
) -> PyResult<(Vec<String>, Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let (gene_list, embedding) = umap_genes(
        py,
        genes,
        ontology,
        similarity,
        groupwise,
        counter,
        distance_transform,
        n_components,
        n_neighbors,
        min_dist,
        random_state,
    )?;

    let (fig, ax_obj) = plot_embedding(
        py,
        embedding.clone_ref(py),
        Some(gene_list.clone()),
        labels,
        title,
        annotate,
        max_labels,
        figsize,
        s,
        alpha,
        ax,
    )?;

    Ok((gene_list, embedding, fig, ax_obj))

}
