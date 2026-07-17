#!/usr/bin/env Rscript
library(optparse)

option_list <- list(
  make_option(
    c("--predictions"),
    type = "character",
    default = "predictions_xgb.csv",
    help = "Predictions file",
    metavar = "character"
  ),
  make_option(
    c("--index"),
    type = "character",
    default = "../../corpora/title/test.arrow",
    help = "index csv or arrow file with column doc_id",
    metavar = "character"
  ),
  make_option(
    c("--ground-truth"),
    type = "character",
    default = "../../corpora/ground-truth.arrow",
    help = "ground-truth.arrow columns idn, kind, uri",
    metavar = "character"
  ),
  make_option(
    c("--res_dir"),
    type = "character",
    default = "eval/",
    help = "Results directory",
    metavar = "character"
  ),
  make_option(
    c("--kind"),
    type = "character",
    default = "title",
    help = "Kind of evaluation",
    metavar = "character"
  ),
  make_option(
    c("--corpus_dir"),
    type = "character",
    default = "../../corpora/title/",
    help = "Corpus directory",
    metavar = "character"
  ),
  make_option(
    c("--n_threads"),
    type = "integer",
    default = 20,
    help = "Number of threads",
    metavar = "integer"
  ),
  make_option(
    c("--set_retrieval_only"),
    type = "logical",
    default = FALSE,
    help = "Set retrieval only mode",
    metavar = "logical"
  )
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)
suppressPackageStartupMessages({
  library(tidyverse)
  library(furrr)
  library(arrow)
})
library(casimir)

if (is.null(opt$res_dir) || is.null(opt$kind) || is.null(opt$corpus_dir)) {
  stop("--res_dir, --kind, and --corpus_dir must be supplied", call. = FALSE)
}

n_bt_global <- 10L
RES_DIR <- opt$res_dir
KIND <- opt$kind
CORPUS_DIR <- opt$corpus_dir
N_THREADS <- opt$n_threads
n_bt_global <- 10L

plan(multicore, workers = as.integer(N_THREADS))

gold_all <- polars::pl$scan_ipc(opt$`ground-truth`) |>
  as_tibble() |>
  filter(kind == KIND)

index <- polars::pl$scan_ipc(opt$index) |>
  as_tibble() |>
  mutate(kind = KIND)

gold <- index |>
  left_join(gold_all, by = c("idn", "kind"), multiple = "all") |>
  transmute(
    doc_id = idn,
    label_id = str_match(uri, "(?<idn>[0-9X]{9,10})$")[, "idn"]
  )

predictions <- read_csv(opt$predictions, show_col_types = FALSE) |>
  mutate(
    label_id = str_match(label_id, "(?<idn>[0-9X]{9,10})$")[, "idn"]
  )

predictions_top5 <- predictions |>
  group_by(doc_id) |>
  mutate(rank = row_number(desc(score))) |>
  filter(rank <= 5)

message("computing set retrieval scores")
res_top5 <- compute_set_retrieval_scores(
  gold_standard = gold,
  seed = 1233415,
  predicted = predictions_top5,
  mode = "doc-avg",
  compute_bootstrap_ci = FALSE,
  n_bt = n_bt_global
)

yaml::as.yaml(res_top5, column.major = FALSE) |>
  str_replace_all(pattern = "- metric: ([a-z1]+)", "\\1@5:") |>
  cat(file = file.path(RES_DIR, "scores_at_5_R.yaml"))

if (opt$set_retrieval_only) {
  quit(save = "no", status = 0)
}

message("computing pr curve")
pr_curve <- compute_pr_curve(
  predicted = predictions,
  gold_standard = gold
)

write_csv(
  select(pr_curve$plot_data, recall = rec, precision = prec_cummax),
  file = file.path(RES_DIR, "pr_curve.csv")
)

message("Computing pr_auc")
pr_auc <- compute_pr_auc_from_curve(pr_curve)

json_output_pr_auc <- pr_auc |>
  transmute(pr_auc) |> #, pr_auc_ci_lower = unname(ci_lower), pr_auc_ci_upper = unname(ci_upper)) |>
  rjson::toJSON() |>
  write_lines(file = file.path(RES_DIR, "pr_auc.json"))

pr_auc_formatted <- pr_auc %>%
  mutate(across(
    c(pr_auc),
    .fns = ~ formatC(
      .x,
      format = "f",
      digits = 3,
      decimal.mark = ",",
      drop0trailing = FALSE
    )
  )) %>%
  mutate(label = paste0("PR-AUC = ", pr_auc))


message("generating plot")
g <- ggplot(pr_curve$plot_data, aes(x = rec, y = prec_cummax)) +
  geom_point() +
  geom_line() +
  ggtitle(
    paste0("Precision-Recall-Kurve ", "-Set"),
    pr_auc_formatted$label[1]
  ) +
  coord_fixed(xlim = c(0, 1)) +
  xlab("Recall") +
  ylab("Precision")

ggsave(
  filename = file.path(RES_DIR, "pr_curve_plot.svg"),
  g,
  device = "svg"
)
