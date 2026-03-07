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
# # 13. Culture Threshold And Concordance Analysis (R)
#
# This notebook performs the analysis side of the culture-versus-metagenomics comparison:
# descriptive Mann-Whitney concordance tests, threshold sweeps from 0% to 10% relative abundance,
# Venn overlap table generation across display cutoffs, and nuisance-adjusted rank-based mixed models.
# It writes all table outputs used by the plotting notebook.
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

                    suppressPackageStartupMessages({
                      library(lmerTest)
                      library(broom.mixed)
                    })


# %% [markdown]
# ## Load Species Abundances, Define Culture Groups, And Compute Descriptive Concordance
#

# %%
culture_groups <- tibble(
  group = c("s_aureus", "p_aeruginosa", "serratia_marcescens", "proteus_mirabilis", "gas", "klebsiella_spp", "e_coli", "acinetobacter_baumannii", "e_faecalis"),
  label = c("S. aureus", "P. aeruginosa", "Serratia", "Proteus", "GAS", "Klebsiella spp.", "E. coli", "A. baumannii", "E. faecalis"),
  culture_col = c("culture_s_aureus", "culture_p_aeruginosa", "culture_serratia_marcescens", "culture_proteus_mirabilis", "culture_gas", "culture_klebsiella_spp", "culture_e_coli", "culture_acinetobacter_baumannii", "culture_e_faecalis"),
  taxa = list(
    c("Staphylococcus aureus"),
    c("Pseudomonas aeruginosa"),
    c("Serratia marcescens"),
    c("Proteus mirabilis"),
    c("Streptococcus pyogenes"),
    c("Klebsiella pneumoniae", "Klebsiella oxytoca"),
    c("Escherichia coli"),
    c("Acinetobacter baumannii"),
    c("Enterococcus faecalis")
  )
)

qc <- read_tsv(table_file(2, "qc_metrics"), show_col_types = FALSE) |>
  mutate(
    culture_date = as.Date(culture_date),
    batch_id = factor(batch_id),
    body_region = factor(body_region, levels = c("lower_extremity", "head_neck", "upper_extremity", "trunk_perineum", "unknown")),
    chronicity_group = factor(chronicity_group, levels = c("unknown", "acute_like", "chronic_like", "mixed")),
    patient_id = factor(sprintf("%02d", as.integer(patient_id)))
  )

counts <- read_csv(file.path(data_dir, "read_count_species_bac.csv"), show_col_types = FALSE)
colnames(counts)[1] <- "sample_id"
counts <- counts |>
  filter(sample_id %in% qc$sample_id)
counts <- counts[match(qc$sample_id, counts$sample_id), ]

count_matrix <- counts |>
  tibble::column_to_rownames("sample_id") |>
  as.matrix()
rel_abundance <- sweep(count_matrix, 1, rowSums(count_matrix), "/")
rel_abundance[is.na(rel_abundance)] <- 0

group_abundance <- tibble(sample_id = rownames(rel_abundance))
for (idx in seq_len(nrow(culture_groups))) {
  taxa_present <- intersect(culture_groups$taxa[[idx]], colnames(rel_abundance))
  values <- if (length(taxa_present) == 0) rep(0, nrow(rel_abundance)) else rowSums(rel_abundance[, taxa_present, drop = FALSE])
  group_abundance[[culture_groups$group[idx]]] <- values
}

culture_model_data <- qc |>
  left_join(group_abundance, by = "sample_id")

descriptive_rows <- vector("list", length = 0)
for (idx in seq_len(nrow(culture_groups))) {
  group_name <- culture_groups$group[idx]
  group_label <- culture_groups$label[idx]
  culture_col <- culture_groups$culture_col[idx]
  observed <- as.logical(culture_model_data[[culture_col]])
  abundance <- culture_model_data[[group_name]]
  positive_values <- abundance[observed]
  negative_values <- abundance[!observed]
  n_positive <- sum(observed, na.rm = TRUE)
  n_negative <- sum(!observed, na.rm = TRUE)
  if (n_positive > 0 && n_negative > 0) {
    test <- suppressWarnings(wilcox.test(positive_values, negative_values, alternative = "greater", exact = FALSE))
    pvalue <- unname(test$p.value)
    u_stat <- unname(test$statistic)
  } else {
    pvalue <- NA_real_
    u_stat <- NA_real_
  }
  descriptive_rows[[length(descriptive_rows) + 1]] <- tibble(
    group = group_name,
    label = group_label,
    n_culture_positive = n_positive,
    n_culture_negative = n_negative,
    u_statistic = u_stat,
    pvalue = pvalue
  )
}
descriptive_concordance <- bind_rows(descriptive_rows) |>
  mutate(qvalue = if_else(!is.na(pvalue), p.adjust(pvalue, method = "BH"), NA_real_))
write_tsv(descriptive_concordance, table_file(42, "culture_concordance_descriptive"))

abundance_plot_df <- culture_model_data |>
  select(sample_id, all_of(culture_groups$culture_col), all_of(culture_groups$group)) |>
  pivot_longer(cols = all_of(culture_groups$group), names_to = "group", values_to = "rel_abundance") |>
  left_join(culture_groups |> select(group, label, culture_col), by = "group") |>
  rowwise() |>
  mutate(culture_status = if_else(as.logical(cur_data()[[culture_col]]), "Culture positive", "Culture negative")) |>
  ungroup() |>
  group_by(group, label, culture_status) |>
  filter(n() > 0) |>
  ungroup() |>
  mutate(log10_rel_abundance = log10(rel_abundance + 1e-6)) |>
  select(sample_id, group, label, culture_status, rel_abundance, log10_rel_abundance)

write_tsv(abundance_plot_df, table_file(43, "culture_abundance_plot_data"))

print(culture_groups)
print(descriptive_concordance)


# %% [markdown]
# ## Sweep Detection Thresholds From 0% To 10%
#

# %%
safe_div <- function(num, den) ifelse(den > 0, num / den, NA_real_)

calc_kappa <- function(tp, fp, fn, tn) {
  total <- tp + fp + fn + tn
  observed <- safe_div(tp + tn, total)
  expected <- safe_div((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn), total^2)
  ifelse(is.na(observed) | is.na(expected) | expected >= 1, NA_real_, (observed - expected) / (1 - expected))
}

thresholds <- seq(0, 0.10, by = 0.001)
threshold_rows <- vector("list", length = 0)

for (idx in seq_len(nrow(culture_groups))) {
  group_name <- culture_groups$group[idx]
  group_label <- culture_groups$label[idx]
  culture_col <- culture_groups$culture_col[idx]

  observed <- as.logical(culture_model_data[[culture_col]])
  abundance <- culture_model_data[[group_name]]

  for (threshold in thresholds) {
    detected <- abundance >= threshold
    tp <- sum(observed & detected, na.rm = TRUE)
    fp <- sum(!observed & detected, na.rm = TRUE)
    fn <- sum(observed & !detected, na.rm = TRUE)
    tn <- sum(!observed & !detected, na.rm = TRUE)

    precision <- safe_div(tp, tp + fp)
    recall <- safe_div(tp, tp + fn)
    f1 <- ifelse(is.na(precision) | is.na(recall) | precision + recall == 0, NA_real_, 2 * precision * recall / (precision + recall))

    threshold_rows[[length(threshold_rows) + 1]] <- tibble(
      group = group_name,
      label = group_label,
      threshold = threshold,
      true_positive = tp,
      false_positive = fp,
      false_negative = fn,
      true_negative = tn,
      sensitivity = recall,
      specificity = safe_div(tn, tn + fp),
      ppv = precision,
      npv = safe_div(tn, tn + fn),
      f1 = f1,
      kappa = calc_kappa(tp, fp, fn, tn)
    )
  }
}

threshold_sweep <- bind_rows(threshold_rows)
optimal_thresholds <- threshold_sweep |>
  group_by(group, label) |>
  arrange(desc(f1), desc(kappa), threshold) |>
  slice_head(n = 1) |>
  ungroup()

display_cutoffs <- c(0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10)
venn_counts <- threshold_sweep |>
  filter(round(threshold, 6) %in% round(display_cutoffs, 6)) |>
  transmute(
    group,
    label,
    threshold,
    culture_only = false_negative,
    sequencing_only = false_positive,
    both = true_positive,
    neither = true_negative
  )

write_tsv(threshold_sweep, table_file(32, "culture_threshold_sweep"))
write_tsv(optimal_thresholds, table_file(33, "culture_optimal_thresholds"))
write_tsv(venn_counts, table_file(35, "culture_venn_counts"))

print(optimal_thresholds)


# %% [markdown]
# ## Fit Technical/Nuisance-Adjusted Rank-Based Mixed Models
#

# %%
concordance_rows <- vector("list", length = 0)

for (idx in seq_len(nrow(culture_groups))) {
  group_name <- culture_groups$group[idx]
  group_label <- culture_groups$label[idx]
  culture_col <- culture_groups$culture_col[idx]
  threshold_used <- optimal_thresholds |>
    filter(group == group_name) |>
    pull(threshold)

  model_df <- culture_model_data |>
    transmute(
      sample_id,
      patient_id,
      batch_id,
      bacterial_species_reads,
      host_removed_fraction,
      log10_bacterial_reads,
      culture_status = factor(
        if_else(as.logical(.data[[culture_col]]), "Culture positive", "Culture negative"),
        levels = c("Culture negative", "Culture positive")
      ),
      rel_abundance = .data[[group_name]]
    ) |>
    filter(
      !is.na(batch_id),
      !is.na(bacterial_species_reads),
      !is.na(host_removed_fraction),
      !is.na(log10_bacterial_reads),
      !is.na(culture_status),
      !is.na(rel_abundance)
    ) |>
    filter(bacterial_species_reads >= 5000) |>
    mutate(
      abundance_rank = rank(rel_abundance, ties.method = "average"),
      normal_score_abundance = qnorm((abundance_rank - 0.5) / n())
    )

  n_positive <- sum(model_df$culture_status == "Culture positive")
  n_negative <- sum(model_df$culture_status == "Culture negative")
  if (n_positive == 0 || n_negative == 0 || sd(model_df$normal_score_abundance) == 0) {
    concordance_rows[[length(concordance_rows) + 1]] <- tibble(
      group = group_name,
      label = group_label,
      threshold = threshold_used,
      n_samples = nrow(model_df),
      n_positive = n_positive,
      n_negative = n_negative,
      predictor = "normal_score_abundance",
      status = "failed_before_fit",
      singular_fit = NA,
      estimate = NA_real_,
      conf.low = NA_real_,
      conf.high = NA_real_,
      pvalue = NA_real_,
      qvalue = NA_real_,
      detail = "No class variation or no variation in the rank-normalized metagenomic abundance response."
    )
    next
  }

  fit_warnings <- character(0)
  fit <- tryCatch(
    withCallingHandlers(
      lmer(
        normal_score_abundance ~ culture_status + host_removed_fraction + log10_bacterial_reads + (1 | patient_id) + (1 | batch_id),
        data = model_df,
        REML = FALSE
      ),
      warning = function(w) {
        fit_warnings <<- c(fit_warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) e
  )

  if (inherits(fit, "error")) {
    concordance_rows[[length(concordance_rows) + 1]] <- tibble(
      group = group_name,
      label = group_label,
      threshold = threshold_used,
      n_samples = nrow(model_df),
      n_positive = n_positive,
      n_negative = n_negative,
      predictor = "normal_score_abundance",
      status = "failed",
      singular_fit = NA,
      estimate = NA_real_,
      conf.low = NA_real_,
      conf.high = NA_real_,
      pvalue = NA_real_,
      qvalue = NA_real_,
      detail = as.character(fit$message)
    )
    next
  }

  abundance_term <- tidy(fit, effects = "fixed", conf.int = TRUE, conf.method = "Wald") |>
    filter(term == "culture_statusCulture positive")
  fit_singular <- isSingular(fit, tol = 1e-5)
  fit_status <- case_when(
    nrow(abundance_term) == 0 || is.na(abundance_term$p.value) || is.na(abundance_term$conf.low) || is.na(abundance_term$conf.high) ~ "separated",
    fit_singular ~ "ok_singular",
    TRUE ~ "ok"
  )
  fit_detail <- c()
  if (fit_status == "separated") {
    fit_detail <- c(fit_detail, "Model showed undefined Wald intervals for the culture-status effect in the rank-based mixed model.")
  }
  if (fit_singular) {
    fit_detail <- c(fit_detail, "Singular random-effects fit (boundary variance estimate).")
  }
  if (length(fit_warnings) > 0) {
    fit_detail <- c(fit_detail, unique(fit_warnings))
  }

  concordance_rows[[length(concordance_rows) + 1]] <- tibble(
    group = group_name,
    label = group_label,
    threshold = threshold_used,
    n_samples = nrow(model_df),
    n_positive = n_positive,
    n_negative = n_negative,
    predictor = "normal_score_abundance",
    status = fit_status,
    singular_fit = fit_singular,
    estimate = abundance_term$estimate,
    conf.low = abundance_term$conf.low,
    conf.high = abundance_term$conf.high,
    pvalue = abundance_term$p.value,
    qvalue = NA_real_,
    detail = paste(unique(fit_detail), collapse = " | ")
  )
}

concordance_table <- bind_rows(concordance_rows)
ok_rows <- concordance_table$status %in% c("ok", "ok_singular") & !is.na(concordance_table$pvalue)
concordance_table$qvalue[ok_rows] <- p.adjust(concordance_table$pvalue[ok_rows], method = "BH")

write_tsv(concordance_table, table_file(34, "culture_mixed_concordance"))

print(concordance_table)


# %% [markdown]
# ## Review Numbered Table Outputs
#

# %%
outputs <- tibble(
  output = c(
    table_file(32, "culture_threshold_sweep"),
    table_file(33, "culture_optimal_thresholds"),
    table_file(34, "culture_mixed_concordance"),
    table_file(35, "culture_venn_counts"),
    table_file(42, "culture_concordance_descriptive"),
    table_file(43, "culture_abundance_plot_data")
  )
)
print(outputs)

