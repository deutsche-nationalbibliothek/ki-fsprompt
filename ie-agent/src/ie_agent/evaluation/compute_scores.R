#!/usr/bin/env Rscript
# Scores of one run with casimir, as in src/compute_metrics.R of KI-FS Prompt (doc-avg, top 100, title gold).
# Differences: the result is one JSON file including the PR-AUC, no plot, and IDs are always read as text
# (read_csv would turn IDs of a small sample into numbers and lose their leading zeros).
suppressPackageStartupMessages({
  library(optparse)
  library(tidyverse)
  library(furrr)
  library(arrow)
  library(casimir)
})

opt <- parse_args(OptionParser(option_list = list(
  make_option("--predictions", type = "character", help = "mapped_predictions.csv of the run"),
  make_option("--index", type = "character", help = "arrow file with column idn: the documents of the run"),
  make_option("--ground-truth", type = "character", help = "ground-truth.arrow with columns idn, kind, uri"),
  make_option("--output", type = "character", help = "scores.json to write"),
  make_option("--kind", type = "character", default = "title", help = "kind of gold standard"),
  make_option("--n_threads", type = "integer", default = 20L, help = "number of threads")
)))

plan(multicore, workers = opt$n_threads)

gold_all <- polars::pl$scan_ipc(opt$`ground-truth`) |>
  as_tibble() |>
  filter(kind == opt$kind)

index <- polars::pl$scan_ipc(opt$index) |>
  as_tibble() |>
  mutate(kind = opt$kind)

gold <- index |>
  left_join(gold_all, by = c("idn", "kind"), multiple = "all") |>
  transmute(
    doc_id = idn,
    label_id = str_match(uri, "(?<idn>[0-9X]{9,10})$")[, "idn"]
  )

predictions <- read_csv(
  opt$predictions,
  col_types = cols(doc_id = col_character(), label_id = col_character()),
  show_col_types = FALSE
) |>
  mutate(label_id = str_match(label_id, "(?<idn>[0-9X]{9,10})$")[, "idn"])

predictions_top100 <- predictions |>
  group_by(doc_id) |>
  mutate(rank = row_number(desc(score))) |>
  filter(rank <= 100)

res_top100 <- compute_set_retrieval_scores(
  gold_standard = gold,
  seed = 1233415,
  predicted = predictions_top100,
  mode = "doc-avg",
  compute_bootstrap_ci = FALSE,
  n_bt = 10L
)

pr_auc <- compute_pr_auc_from_curve(compute_pr_curve(predicted = predictions, gold_standard = gold))

scores <- list()
for (i in seq_len(nrow(res_top100))) {
  scores[[paste0(res_top100$metric[i], "@100")]] <- list(
    mode = res_top100$mode[i], value = res_top100$value[i], support = res_top100$support[i]
  )
}
scores$pr_auc <- pr_auc$pr_auc[1]
write_lines(rjson::toJSON(scores, indent = 2), opt$output)
