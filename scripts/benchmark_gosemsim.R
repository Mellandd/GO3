args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx)) return(default)
  if (idx == length(args)) stop(paste("Missing value for", flag))
  args[[idx + 1]]
}

read_linux_mem_mb <- function(field_name) {
  if (!file.exists("/proc/self/status")) return(NA_real_)
  lines <- readLines("/proc/self/status", warn = FALSE)
  target <- lines[startsWith(lines, field_name)]
  if (length(target) == 0) return(NA_real_)
  # Example: "VmRSS:\t  123456 kB"
  num <- as.numeric(gsub("[^0-9]", "", target[[1]]))
  if (is.na(num)) return(NA_real_)
  num / 1024.0
}

mode <- get_arg("--mode", "term")
ontology <- get_arg("--ontology", "BP")
measure <- get_arg("--measure", "Resnik")
pairs_tsv <- get_arg("--pairs-tsv", NULL)
orgdb <- get_arg("--orgdb", "org.Hs.eg.db")
anno_tsv <- get_arg("--anno-tsv", NULL)
warmup <- as.integer(get_arg("--warmup", "1"))
repeats <- as.integer(get_arg("--repeats", "3"))
seed <- as.integer(get_arg("--seed", "42"))

if (!ontology %in% c("BP", "MF", "CC")) {
  stop("Invalid ontology; expected BP, MF, or CC")
}

missing_pkgs <- c()
if (!suppressWarnings(requireNamespace("Rcpp", quietly = TRUE))) {
  missing_pkgs <- c(missing_pkgs, "Rcpp")
}
if (!suppressWarnings(requireNamespace("GOSemSim", quietly = TRUE))) {
  missing_pkgs <- c(missing_pkgs, "GOSemSim")
}
if (is.null(anno_tsv) || anno_tsv == "") {
  if (!suppressWarnings(requireNamespace(orgdb, quietly = TRUE))) {
    missing_pkgs <- c(missing_pkgs, orgdb)
  }
}
if (length(missing_pkgs) > 0) {
  missing_str <- paste(missing_pkgs, collapse = ", ")
  install_str <- paste0("BiocManager::install(c(", paste(sprintf("'%s'", missing_pkgs), collapse = ", "), "))")
  stop(paste0("Missing R package(s): ", missing_str, " | install with: ", install_str))
}

suppressPackageStartupMessages({
  # Force-load Rcpp to make C-callables available for IC-based methods
  # (Lin/Resnik/Jiang/Rel) used internally by GOSemSim.
  library(Rcpp)
  library(GOSemSim)
})

set.seed(seed)

build_semdata <- function() {
  if (!is.null(anno_tsv) && anno_tsv != "") {
    if (!file.exists(anno_tsv)) {
      stop(paste("anno TSV not found:", anno_tsv))
    }
    goAnno <- read.delim(anno_tsv, header = TRUE, sep = "\t", stringsAsFactors = FALSE, quote = "", comment.char = "")
    if (!all(c("GO", "ONTOLOGY") %in% names(goAnno))) {
      stop("anno TSV must include GO and ONTOLOGY columns")
    }
    return(godata(annoDb = goAnno, ont = ontology, computeIC = TRUE))
  }
  return(godata(orgdb, ont = ontology, computeIC = TRUE))
}

if (mode == "loading") {
  t0 <- proc.time()[[3]]
  semData <- tryCatch(build_semdata(), error = function(e) {
    stop(paste("Failed to build semData:", conditionMessage(e)))
  })
  t1 <- proc.time()[[3]]

  payload <- sprintf(
    '{"lib":"gosemsim","total_time_s":%.6f,"peak_rss_mb":%.3f,"final_rss_mb":%.3f,"display_name":"GOSemSim","details":{"ontology":"%s","measure":"%s"}}',
    (t1 - t0),
    read_linux_mem_mb("VmHWM:"),
    read_linux_mem_mb("VmRSS:"),
    ontology,
    measure
  )
  cat(payload)
  quit(save = "no", status = 0)
}

if (is.null(pairs_tsv)) {
  stop("--pairs-tsv is required for term/gene benchmark modes")
}
if (!file.exists(pairs_tsv)) {
  stop(paste("pairs TSV not found:", pairs_tsv))
}

semData <- tryCatch(build_semdata(), error = function(e) {
  stop(paste("Failed to build semData:", conditionMessage(e)))
})

df <- read.delim(
  pairs_tsv,
  header = FALSE,
  sep = "\t",
  stringsAsFactors = FALSE,
  quote = "",
  comment.char = ""
)
if (ncol(df) < 3) {
  stop("pairs TSV must contain at least 3 columns")
}
colnames(df)[1:3] <- c("size", "a", "b")
sizes <- sort(unique(df$size))

parse_terms <- function(s) {
  if (is.na(s) || s == "") return(character(0))
  pieces <- strsplit(s, ",", fixed = TRUE)[[1]]
  pieces <- trimws(pieces)
  pieces <- pieces[pieces != ""]
  unique(pieces)
}

safe_go_sim <- function(go1, go2) {
  v <- goSim(go1, go2, semData, measure = measure)
  if (is.na(v)) return(0.0)
  v
}

safe_mgo_sim <- function(go_vec1, go_vec2) {
  if (length(go_vec1) == 0 || length(go_vec2) == 0) return(0.0)
  v <- mgoSim(go_vec1, go_vec2, semData, measure = measure, combine = "BMA")
  if (is.na(v)) return(0.0)
  v
}

run_once <- function(sub) {
  if (mode == "term") {
    out <- mapply(safe_go_sim, sub$a, sub$b)
    invisible(out)
    return(invisible(NULL))
  }
  if (mode == "gene") {
    out <- mapply(
      function(x, y) safe_mgo_sim(parse_terms(x), parse_terms(y)),
      sub$a,
      sub$b
    )
    invisible(out)
    return(invisible(NULL))
  }
  stop(paste("Unknown mode:", mode))
}

if (length(sizes) == 0) {
  stop("No rows found in pairs TSV")
}

if (warmup > 0) {
  sub0 <- df[df$size == sizes[[1]], ]
  for (i in seq_len(warmup)) {
    run_once(sub0)
  }
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

write.table(results, file = "", sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
