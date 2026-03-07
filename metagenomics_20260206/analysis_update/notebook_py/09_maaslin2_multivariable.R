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
# # 09. MaAsLin2 Multivariable Associations
#
# This notebook runs MaAsLin2 as an established multivariable microbiome association method.
# The analysis is kept patient-aware through a patient random effect, but the feature space is trimmed to the
# most prevalent taxa so the resulting table stays interpretable.
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

library(Maaslin2)


# %% [markdown]
# ## Load QC-Passing Data And Prefilter The Feature Space
#
# The MaAsLin2 run starts from the Bracken bacterial table on the model-QC-passing samples.
# To avoid flooding the results with extremely sparse long-tail species, the notebook keeps taxa with at least
# 10% prevalence and then limits the model to the 200 most abundant among those.
#

# %%
qc <- read_tsv(table_file(2, "qc_metrics"), show_col_types = FALSE) |>
  mutate(model_qc_pass = tolower(as.character(model_qc_pass)) %in% c("true", "t", "1"))

counts <- read_csv(file.path(data_dir, "read_count_species_bac.csv"), show_col_types = FALSE)
colnames(counts)[1] <- "sample_id"

model_samples <- qc |>
  filter(model_qc_pass) |>
  select(sample_id, patient_id, body_region, chronicity_group, clinical_infection_flag, log10_bacterial_reads)

counts <- counts |>
  filter(sample_id %in% model_samples$sample_id)
counts <- counts[match(model_samples$sample_id, counts$sample_id), ]

count_matrix <- counts |>
  tibble::column_to_rownames("sample_id") |>
  as.data.frame(check.names = FALSE)

rel_abundance <- sweep(as.matrix(count_matrix), 1, rowSums(count_matrix), "/")
rel_abundance[is.na(rel_abundance)] <- 0

feature_summary <- tibble(
  feature = colnames(count_matrix),
  prevalence = colMeans(count_matrix > 0),
  mean_relative_abundance = colMeans(rel_abundance)
) |>
  filter(prevalence >= 0.1) |>
  arrange(desc(mean_relative_abundance))

selected_features <- head(feature_summary$feature, 200)
count_matrix <- as.data.frame(count_matrix[, selected_features, drop = FALSE], check.names = FALSE)
rownames(count_matrix) <- model_samples$sample_id

model_metadata <- model_samples |>
  mutate(
    patient_id = factor(sprintf("%02d", as.integer(patient_id))),
    body_region = factor(body_region),
    chronicity_group = factor(chronicity_group),
    clinical_infection_flag = factor(clinical_infection_flag)
  ) |>
  tibble::column_to_rownames("sample_id") |>
  as.data.frame()

maaslin_summary <- tibble(
  n_samples = nrow(model_metadata),
  n_prevalent_features = nrow(feature_summary),
  n_features_tested = ncol(count_matrix),
  fixed_effects = "body_region + chronicity_group + clinical_infection_flag + log10_bacterial_reads",
  random_effects = "patient_id"
)

write_tsv(maaslin_summary, table_file(23, "maaslin2_model_summary"))

print(maaslin_summary)
print(feature_summary |> slice_head(n = 15))


# %% [markdown]
# ## Run MaAsLin2
#
# The model uses total-sum scaling plus log transformation, fixed effects for body region, chronicity, infection flag,
# and bacterial depth, and a patient random effect.
#

# %%
maaslin_dir <- file.path(root, "maaslin2")
unlink(maaslin_dir, recursive = TRUE)
dir.create(maaslin_dir, recursive = TRUE, showWarnings = FALSE)

fit <- Maaslin2(
  input_data = count_matrix,
  input_metadata = model_metadata,
  output = maaslin_dir,
  min_prevalence = 0.1,
  normalization = "TSS",
  transform = "LOG",
  analysis_method = "LM",
  max_significance = 0.1,
  fixed_effects = c("body_region", "chronicity_group", "clinical_infection_flag", "log10_bacterial_reads"),
  random_effects = c("patient_id"),
  reference = c(
    "body_region,lower_extremity",
    "chronicity_group,unknown",
    "clinical_infection_flag,unknown"
  ),
  plot_heatmap = FALSE,
  plot_scatter = FALSE,
  cores = 1
)

all_results <- read_tsv(file.path(maaslin_dir, "all_results.tsv"), show_col_types = FALSE) |>
  mutate(
    feature_label = str_squish(str_replace_all(feature, "\\.+", " ")),
    term_label = if_else(is.na(value) | value == "", metadata, paste(metadata, value, sep = " = ")),
    conf_low = coef - 1.96 * stderr,
    conf_high = coef + 1.96 * stderr
  ) |>
  arrange(qval, pval)

focus_taxa <- c(
  "Staphylococcus aureus",
  "Pseudomonas aeruginosa",
  "Cutibacterium acnes",
  "Serratia marcescens",
  "Klebsiella pneumoniae",
  "Corynebacterium striatum",
  "Escherichia coli",
  "Proteus mirabilis",
  "Acinetobacter baumannii",
  "Enterococcus faecalis"
)

focus_results <- all_results |>
  filter(feature_label %in% focus_taxa) |>
  arrange(qval, pval)

write_tsv(focus_results, table_file(24, "maaslin2_focus_results"))

print(focus_results |> slice_head(n = 25))


# %% [markdown]
# ## Summarize Positive And Negative Results
#
# The figure focuses on the clinically relevant taxa that overlap the earlier regression and mixed-model analyses.
#

# %%
plot_df <- focus_results |>
  filter(term_label != "patient_id") |>
  arrange(qval, desc(abs(coef))) |>
  slice_head(n = 20) |>
  mutate(
    significant = qval <= 0.1,
    plot_label = paste(feature_label, term_label, sep = " | ")
  )

if (nrow(plot_df) == 0) {
  plot_df <- all_results |>
    arrange(qval, pval) |>
    slice_head(n = 20) |>
    mutate(
      significant = qval <= 0.1,
      plot_label = paste(feature_label, term_label, sep = " | ")
    )
}

plot_df <- plot_df |>
  mutate(plot_label = factor(plot_label, levels = rev(unique(plot_label))))

figure_09 <- ggplot(plot_df, aes(x = coef, y = plot_label, color = significant)) +
  geom_vline(xintercept = 0, linewidth = 0.4, linetype = "dashed") +
  geom_errorbarh(aes(xmin = conf_low, xmax = conf_high), height = 0.2, linewidth = 0.5) +
  geom_point(size = 2.4) +
  scale_color_manual(values = c("TRUE" = "#b44b2a", "FALSE" = "#4c6c8a")) +
  labs(
    title = "MaAsLin2 Focused Association Summary",
    x = "Coefficient estimate",
    y = NULL,
    color = "q <= 0.1"
  ) +
  theme(legend.position = "top")

ggsave(
  figure_file(9, "maaslin2_summary"),
  figure_09,
  width = 13,
  height = 9,
  device = grDevices::svg
)
print(figure_09)

maaslin_findings <- tibble(
  finding = c(
    sprintf("Positive result: %d of %d tested feature-term combinations reached q <= 0.1.", sum(all_results$qval <= 0.1, na.rm = TRUE), nrow(all_results)),
    sprintf("Positive result: %d focus-taxon terms reached q <= 0.1.", sum(focus_results$qval <= 0.1, na.rm = TRUE)),
    "Negative result: many significant hits fall in correlated long-tail species neighborhoods, so the focus figure is used to keep interpretation clinically grounded."
  )
)

print(maaslin_findings)

