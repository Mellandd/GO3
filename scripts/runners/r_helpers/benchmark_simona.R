# Benchmark helper for simona (Bioconductor).
#
# Modes (passed via --mode):
#   loading  : build the DAG + IC and emit a JSON loading payload
#   term     : read a TSV (size<TAB>a<TAB>b) and time goSim per group
#   gene     : read a TSV (size<TAB>terms1_csv<TAB>terms2_csv) and time
#              group_sim BMA per group
#
# Methods supported via --measure:
#   Resnik | Lin | Wang   (mapped to simona's Sim_*_* method names)
#
# Annotation source: an --anno-tsv with columns gene<TAB>GO<TAB>ONTOLOGY
# (same TSV the gosemsim helper reads). simona doesn't parse GAF natively,
# so the orchestrator writes this TSV from the user's GAF.

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx)) return(default)
  if (idx == length(args)) stop(paste("Missing value for", flag))
  args[[idx + 1]]
}

mode <- get_arg("--mode", "term")
ontology <- get_arg("--ontology", "BP")
measure <- get_arg("--measure", "Resnik")
obo_path <- get_arg("--obo", NULL)
anno_tsv <- get_arg("--anno-tsv", NULL)
pairs_tsv <- get_arg("--pairs-tsv", NULL)
warmup <- as.integer(get_arg("--warmup", "1"))
repeats <- as.integer(get_arg("--repeats", "3"))
seed <- as.integer(get_arg("--seed", "42"))

if (!ontology %in% c("BP", "MF", "CC")) {
  stop("Invalid ontology; expected BP, MF, or CC")
}
if (is.null(obo_path)) stop("--obo is required")
if (is.null(anno_tsv)) stop("--anno-tsv is required")

method_map <- list(
  Resnik = "Sim_Resnik_1999",
  Lin    = "Sim_Lin_1998",
  Wang   = "Sim_Wang_2007"
)
sim_method <- method_map[[measure]]
if (is.null(sim_method)) {
  stop(paste("Unsupported measure for simona:", measure))
}

if (!suppressWarnings(requireNamespace("simona", quietly = TRUE))) {
  stop("Missing R package 'simona'. Install with: BiocManager::install('simona')")
}
suppressPackageStartupMessages(library(simona))

set.seed(seed)

read_anno <- function(path) {
  df <- read.delim(path, header = TRUE, sep = "\t", stringsAsFactors = FALSE,
                   quote = "", comment.char = "")
  if (!all(c("gene", "GO", "ONTOLOGY") %in% names(df))) {
    stop("anno TSV must have columns gene, GO, ONTOLOGY")
  }
  df <- df[df$ONTOLOGY == ontology, , drop = FALSE]
  df
}

build_gene2terms <- function(df) {
  split(df$GO, df$gene)
}

build_term2genes <- function(df) {
  split(df$gene, df$GO)
}

build_dag <- function() {
  df <- read_anno(anno_tsv)
  t2g <- build_term2genes(df)
  # simona's import_obo signature: (file, relation_type, inherit_relations, verbose, ...)
  # `relation_type` selects which non-is_a relations to keep; use `part_of`.
  dag <- import_obo(obo_path, relation_type = "part_of")
  # Attach gene -> term annotation post-hoc. In current simona this is
  # `add_annotation(dag, annotation = list(term -> genes))`.
  annotate_called <- FALSE
  if (exists("add_annotation", where = asNamespace("simona"), inherits = FALSE)) {
    dag <- simona::add_annotation(dag, annotation = t2g)
    annotate_called <- TRUE
  } else if (exists("dag_annotate", where = asNamespace("simona"), inherits = FALSE)) {
    dag <- simona::dag_annotate(dag, annotation = t2g)
    annotate_called <- TRUE
  }
  if (!annotate_called) {
    warning("Could not attach annotation to simona DAG; IC-based methods may fail.")
  }
  list(dag = dag, gene2terms = build_gene2terms(df))
}

if (mode == "loading") {
  t0 <- proc.time()[[3]]
  built <- tryCatch(build_dag(), error = function(e) {
    stop(paste("Failed to build simona DAG:", conditionMessage(e)))
  })
  t1 <- proc.time()[[3]]

  # peak_rss_mb is filled in by the Python parent via wait4/rusage
  # (cross-platform). final_rss_mb is not reliably measurable from the
  # outside, so it is emitted as null.
  payload <- sprintf(
    paste0(
      '{"lib":"simona","total_time_s":%.6f,"peak_rss_mb":null,',
      '"final_rss_mb":null,"display_name":"simona",',
      '"details":{"ontology":"%s","measure":"%s","method":"%s"}}'
    ),
    (t1 - t0),
    ontology,
    measure,
    sim_method
  )
  cat(payload)
  quit(save = "no", status = 0)
}

# `scores` mode: compute similarity for every pair in `pairs_tsv` (order
# preserved) and emit a single-column TSV (`score`). Used by
# scripts/validate_cross_tool.py to align scores across libraries.
if (mode == "scores-term" || mode == "scores-gene") {
  if (is.null(pairs_tsv)) stop("--pairs-tsv is required for scores modes")
  if (!file.exists(pairs_tsv)) stop(paste("pairs TSV not found:", pairs_tsv))

  built <- build_dag()
  dag <- built$dag

  df <- read.delim(pairs_tsv, header = FALSE, sep = "\t",
                   stringsAsFactors = FALSE, quote = "", comment.char = "")
  if (ncol(df) < 3) stop("pairs TSV must contain at least 3 columns")
  colnames(df)[1:3] <- c("size", "a", "b")

  parse_terms <- function(s) {
    if (is.na(s) || s == "") return(character(0))
    pieces <- strsplit(s, ",", fixed = TRUE)[[1]]
    pieces <- trimws(pieces)
    pieces <- pieces[pieces != ""]
    unique(pieces)
  }

  safe_term_sim <- function(a, b) {
    v <- tryCatch(
      term_sim(dag, c(a, b), method = sim_method)[1, 2],
      error = function(e) NA_real_
    )
    if (is.na(v)) 0.0 else as.numeric(v)
  }

  safe_group_sim <- function(g1, g2) {
    if (length(g1) == 0 || length(g2) == 0) return(0.0)
    v <- tryCatch(
      group_sim(dag, g1, g2,
                method = "GroupSim_pairwise_BMA",
                control = list(term_sim_method = sim_method)),
      error = function(e) NA_real_
    )
    if (is.na(v)) 0.0 else as.numeric(v)
  }

  if (mode == "scores-term") {
    out <- mapply(safe_term_sim, df$a, df$b)
  } else {
    out <- mapply(
      function(x, y) safe_group_sim(parse_terms(x), parse_terms(y)),
      df$a, df$b
    )
  }
  res <- data.frame(score = as.numeric(out))
  write.table(res, file = "", sep = "\t", row.names = FALSE,
              col.names = TRUE, quote = FALSE)
  quit(save = "no", status = 0)
}

if (is.null(pairs_tsv)) stop("--pairs-tsv is required for term/gene benchmark modes")
if (!file.exists(pairs_tsv)) stop(paste("pairs TSV not found:", pairs_tsv))

built <- build_dag()
dag <- built$dag

df <- read.delim(pairs_tsv, header = FALSE, sep = "\t", stringsAsFactors = FALSE,
                 quote = "", comment.char = "")
if (ncol(df) < 3) stop("pairs TSV must contain at least 3 columns")
colnames(df)[1:3] <- c("size", "a", "b")
sizes <- sort(unique(df$size))

parse_terms <- function(s) {
  if (is.na(s) || s == "") return(character(0))
  pieces <- strsplit(s, ",", fixed = TRUE)[[1]]
  pieces <- trimws(pieces)
  pieces <- pieces[pieces != ""]
  unique(pieces)
}

safe_term_sim <- function(a, b) {
  v <- tryCatch(
    term_sim(dag, c(a, b), method = sim_method)[1, 2],
    error = function(e) NA_real_
  )
  if (is.na(v)) 0.0 else as.numeric(v)
}

safe_group_sim <- function(g1, g2) {
  if (length(g1) == 0 || length(g2) == 0) return(0.0)
  v <- tryCatch(
    group_sim(dag, g1, g2,
              method = "GroupSim_pairwise_BMA",
              control = list(term_sim_method = sim_method)),
    error = function(e) NA_real_
  )
  if (is.na(v)) 0.0 else as.numeric(v)
}

run_once <- function(sub) {
  if (mode == "term") {
    out <- mapply(safe_term_sim, sub$a, sub$b)
    invisible(out)
    return(invisible(NULL))
  }
  if (mode == "gene") {
    out <- mapply(
      function(x, y) safe_group_sim(parse_terms(x), parse_terms(y)),
      sub$a, sub$b
    )
    invisible(out)
    return(invisible(NULL))
  }
  stop(paste("Unknown mode:", mode))
}

if (length(sizes) == 0) stop("No rows found in pairs TSV")

if (warmup > 0) {
  sub0 <- df[df$size == sizes[[1]], ]
  for (i in seq_len(warmup)) run_once(sub0)
}

results <- data.frame(size = integer(), median_s = numeric())
for (sz in sizes) {
  sub <- df[df$size == sz, ]
  times <- numeric()
  for (r in seq_len(max(1L, repeats))) {
    t0 <- proc.time()[[3]]
    run_once(sub)
    t1 <- proc.time()[[3]]
    times <- c(times, t1 - t0)
  }
  results <- rbind(results, data.frame(size = as.integer(sz), median_s = median(times)))
}

write.table(results, file = "", sep = "\t", row.names = FALSE,
            col.names = TRUE, quote = FALSE)
