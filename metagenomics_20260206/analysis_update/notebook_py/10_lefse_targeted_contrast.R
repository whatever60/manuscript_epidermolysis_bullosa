# ---
# jupyter:
#   jupytext:
#     formats: ipynb,notebook_py//R:percent
#     text_representation:
#       extension: .R
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: R (eb)
#     language: R
#     name: ir-eb
# ---

# %% [markdown]
# # 10. LEfSe Targeted Contrast
#
# This notebook uses LEfSe for a narrower exploratory contrast rather than a full multivariable analysis.
# The chosen comparison is `head_neck` versus `lower_extremity`, which was one of the clearer site patterns in the earlier models.
#

# %%
options(width = 140)
suppressPackageStartupMessages({
    library(readr)
    library(dplyr)
    library(tidyr)
    library(ggplot2)
    library(stringr)
})

theme_set(theme_bw(base_size = 12))

root <- getwd()
data_dir <- dirname(root)
figure_dir <- file.path(root, "figures")
table_dir <- file.path(root, "tables")
dir.create(figure_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(table_dir, showWarnings = FALSE, recursive = TRUE)
species_bac_path <- file.path(table_dir, "table_00_02_read_count_species_bac.csv")
if (!file.exists(species_bac_path)) {
  species_bac_path <- file.path(data_dir, "read_count_species_bac.csv")
}

table_id_map <- c(
  `1` = "01_01", `2` = "02_01", `3` = "02_02", `4` = "03_01", `5` = "03_02",
  `6` = "04_01", `7` = "05_01", `8` = "06_01", `9` = "06_02", `10` = "06_03",
  `11` = "06_04", `12` = "06_05", `13` = "06_06", `14` = "07_01", `15` = "07_02",
  `16` = "07_03", `17` = "07_04", `18` = "07_05", `19` = "07_06", `20` = "08_01",
  `21` = "08_02", `22` = "08_03", `23` = "09_01", `24` = "09_02", `25` = "10_01",
  `26` = "10_02", `27` = "11_01", `28` = "11_02", `29` = "12_01", `30` = "12_02",
  `31` = "12_03", `32` = "13_01", `33` = "13_02", `34` = "13_03", `35` = "13_04",
  `36` = "14_01", `37` = "14_02", `38` = "12_04", `39` = "12_05", `40` = "12_06",
  `41` = "12_07", `42` = "13_05", `43` = "13_06"
)

figure_id_map <- c(
  `1` = "02_01", `2` = "03_01", `3` = "04_01", `4` = "05_01", `5` = "06_01",
  `6` = "06_02", `7` = "07_01", `8` = "08_01", `9` = "09_01", `10` = "10_01",
  `11` = "11_01", `12` = "12_01", `13` = "12_02", `14` = "13_01", `15` = "13_02",
  `16` = "13_03", `17` = "13_04", `18` = "14_01", `19` = "14_02", `20` = "14_03",
  `21` = "14_04", `22` = "12_03", `23` = "12_04", `24` = "12_05"
)

table_file <- function(number, slug) {
  prefix <- table_id_map[[as.character(number)]]
  file.path(table_dir, sprintf("table_%s_%s.tsv", prefix, slug))
}

figure_file <- function(number, slug) {
  prefix <- figure_id_map[[as.character(number)]]
  file.path(figure_dir, sprintf("fig_%s_%s.svg", prefix, slug))
}


# %% [markdown]
# ## Build A Targeted LEfSe Input Table
#
# LEfSe is applied to a two-class body-region contrast on the model-QC-passing samples.
# The input keeps taxa with at least 20% prevalence in this subset and then limits the analysis to the top 80 by mean relative abundance.
#

# %%
qc <- read_tsv(table_file(2, "qc_metrics"), show_col_types = FALSE) |>
  mutate(model_qc_pass = tolower(as.character(model_qc_pass)) %in% c("true", "t", "1"))

counts <- read_csv(species_bac_path, show_col_types = FALSE)
colnames(counts)[1] <- "sample_id"

lefse_samples <- qc |>
  filter(model_qc_pass, body_region %in% c("head_neck", "lower_extremity")) |>
  select(sample_id, patient_id, body_region)

counts <- counts |>
  filter(sample_id %in% lefse_samples$sample_id)
counts <- counts[match(lefse_samples$sample_id, counts$sample_id), ]

count_matrix <- counts |>
  tibble::column_to_rownames("sample_id") |>
  as.matrix()

rel_abundance <- sweep(count_matrix, 1, rowSums(count_matrix), "/")
rel_abundance[is.na(rel_abundance)] <- 0

feature_summary <- tibble(
  feature = colnames(rel_abundance),
  prevalence = colMeans(rel_abundance > 0),
  mean_relative_abundance = colMeans(rel_abundance)
) |>
  filter(prevalence >= 0.2) |>
  arrange(desc(mean_relative_abundance))

selected_features <- head(feature_summary$feature, 80)

lefse_input <- bind_cols(
  lefse_samples |>
    transmute(
      class = body_region,
      subclass = "all",
      subject = sprintf("%02d", as.integer(patient_id))
    ),
  as_tibble(rel_abundance[, selected_features, drop = FALSE])
)

lefse_dir <- file.path(root, "lefse")
input_dir <- file.path(lefse_dir, "inputs")
output_dir <- file.path(lefse_dir, "output")
unlink(lefse_dir, recursive = TRUE)
dir.create(input_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

lefse_input_path <- file.path(input_dir, "lefse_body_region_columns.tsv")
write_tsv(lefse_input, lefse_input_path)

lefse_summary <- tibble(
  comparison = "head_neck_vs_lower_extremity",
  n_samples = nrow(lefse_samples),
  n_head_neck = sum(lefse_samples$body_region == "head_neck"),
  n_lower_extremity = sum(lefse_samples$body_region == "lower_extremity"),
  n_prevalent_features = nrow(feature_summary),
  n_features_tested = length(selected_features)
)

write_tsv(lefse_summary, table_file(25, "lefse_comparison_summary"))

print(lefse_summary)
print(feature_summary |> slice_head(n = 15))


# %% [markdown]
# ## Run LEfSe
#
# The notebook calls the `lefse_format_input.py` and `lefse_run.py` executables from the active `eb` environment.
#

# %%
env_prefix <- dirname(dirname(R.home()))
env_bin <- file.path(env_prefix, "bin")
lefse_format_exe <- file.path(env_bin, "lefse_format_input.py")
lefse_run_exe <- file.path(env_bin, "lefse_run.py")

lefse_formatted_path <- file.path(output_dir, "lefse_body_region.in")
lefse_result_path <- file.path(output_dir, "lefse_body_region.res")

format_log <- system2(
  lefse_format_exe,
  args = c(lefse_input_path, lefse_formatted_path, "-f", "c", "-c", "1", "-s", "2", "-u", "3"),
  stdout = TRUE,
  stderr = TRUE
)
if (!is.null(attr(format_log, "status"))) {
  stop(paste(format_log, collapse = "\n"))
}

run_log <- system2(
  lefse_run_exe,
  args = c(lefse_formatted_path, lefse_result_path, "-l", "2.0"),
  stdout = TRUE,
  stderr = TRUE
)
if (!is.null(attr(run_log, "status"))) {
  stop(paste(run_log, collapse = "\n"))
}

raw_results <- read.delim(
  lefse_result_path,
  sep = "\t",
  header = FALSE,
  fill = TRUE,
  stringsAsFactors = FALSE
)
colnames(raw_results) <- c("feature", "log10_mean_abundance", "enriched_group", "lda_score", "wilcoxon_pvalue")

lefse_results <- raw_results |>
  mutate(
    feature_label = str_squish(str_replace_all(feature, "\\.+", " ")),
    log10_mean_abundance = suppressWarnings(as.numeric(log10_mean_abundance)),
    lda_score = suppressWarnings(as.numeric(lda_score)),
    wilcoxon_pvalue = suppressWarnings(as.numeric(wilcoxon_pvalue))
  ) |>
  filter(feature_label != "subject")

significant_results <- lefse_results |>
  filter(!is.na(lda_score), abs(lda_score) >= 2, !is.na(enriched_group), enriched_group != "-") |>
  arrange(desc(abs(lda_score)))

write_tsv(significant_results, table_file(26, "lefse_significant_features"))

print(significant_results |> slice_head(n = 25))


# %% [markdown]
# ## Summarize Positive And Negative Results
#
# LEfSe is intentionally treated as an exploratory targeted contrast. It does not replace the patient-aware multivariable models.
#

# %%
if (nrow(significant_results) > 0) {
  plot_df <- significant_results |>
    slice_head(n = 20) |>
    mutate(feature_label = factor(feature_label, levels = rev(unique(feature_label))))

  figure_10 <- ggplot(plot_df, aes(x = lda_score, y = feature_label, fill = enriched_group)) +
    geom_col(width = 0.75) +
    labs(
      title = "LEfSe Head/Neck Versus Lower Extremity",
      x = "LDA score",
      y = NULL,
      fill = "Enriched in"
    )
} else {
  figure_10 <- ggplot() +
    annotate("text", x = 0, y = 0, label = "No features exceeded the LDA threshold in this targeted contrast.", size = 5) +
    xlim(-1, 1) +
    ylim(-1, 1) +
    theme_void() +
    labs(title = "LEfSe Head/Neck Versus Lower Extremity")
}

ggsave(
  figure_file(10, "lefse_summary"),
  figure_10,
  width = 11,
  height = 8,
  device = grDevices::svg
)
print(figure_10)

lefse_findings <- tibble(
  finding = c(
    sprintf("Positive result: %d features exceeded the LEfSe LDA threshold in the targeted site comparison.", nrow(significant_results)),
    "Negative result: LEfSe is being used on a two-class targeted contrast here because the broader multiclass setup was too brittle and returned an all-null result.",
    "Negative result: LEfSe does not model the full repeated-measures multivariable structure, so its output stays exploratory."
  )
)

print(lefse_findings)
