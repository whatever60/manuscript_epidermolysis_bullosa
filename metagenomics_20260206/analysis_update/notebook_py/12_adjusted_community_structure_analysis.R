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
# # 12. Adjusted Community Structure Analysis (R)
#
# This notebook revisits community similarity with multivariable models.
# It combines a patient-aware PERMANOVA for overall community structure with pairwise mixed models
# that ask whether shared patient or shared body site remain associated with lower Bray-Curtis distance
# after adjusting for technical batch, host burden, read depth, and patient-relative elapsed time.
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

                    suppressPackageStartupMessages({
                      library(vegan)
                      library(lme4)
                      library(lmerTest)
                      library(broom.mixed)
                      library(emmeans)
                    })


# %% [markdown]
# ## Load QC-Passing Community Data
#
# The community models use bacterial relative abundance on the model-QC-passing samples.
# The fixed-effect set is deliberately limited to avoid throwing in several collinear depth-like covariates at once.
#

# %%
qc <- read_tsv(table_file(2, "qc_metrics"), show_col_types = FALSE) |>
  mutate(
    model_qc_pass = tolower(as.character(model_qc_pass)) %in% c("true", "t", "1"),
    culture_date = as.Date(culture_date),
    batch_id = factor(batch_id),
    culture_positive = factor(if_else(as.logical(culture_positive), "positive", "negative"),
                              levels = c("negative", "positive")),
    patient_id = factor(sprintf("%02d", as.integer(patient_id))),
    body_region = factor(body_region, levels = c("lower_extremity", "head_neck", "upper_extremity", "trunk_perineum", "others")),
    chronicity_group = factor(chronicity_group, levels = c("unknown", "acute_like", "chronic_like", "mixed"))
  ) |>
  filter(model_qc_pass) |>
  select(
    sample_id,
    patient_id,
    batch_id,
    culture_date,
    years_since_first_sample,
    location,
    body_region,
    chronicity_group,
    culture_positive,
    host_removed_fraction,
    log10_bacterial_reads
  )

counts <- read_csv(species_bac_path, show_col_types = FALSE)
colnames(counts)[1] <- "sample_id"
counts <- counts |>
  filter(sample_id %in% qc$sample_id)
counts <- counts[match(qc$sample_id, counts$sample_id), ]

count_matrix <- counts |>
  tibble::column_to_rownames("sample_id") |>
  as.matrix()

rel_abundance <- sweep(count_matrix, 1, rowSums(count_matrix), "/")
rel_abundance[is.na(rel_abundance)] <- 0

community_summary <- tibble(
  n_samples = nrow(qc),
  n_patients = n_distinct(qc$patient_id),
  n_batches = n_distinct(qc$batch_id),
  n_species = ncol(rel_abundance),
  fixed_effects = "batch_id + host_removed_fraction + log10_bacterial_reads + years_since_first_sample + body_region + chronicity_group + culture_positive",
  patient_handling = "adonis2 strata = patient_id; pairwise lmer with sample-level random intercepts"
)

print(community_summary)


# %% [markdown]
# ## Run A Patient-Restricted PERMANOVA
#

# %%
set.seed(20260303)
permanova <- adonis2(
  rel_abundance ~ batch_id + host_removed_fraction + log10_bacterial_reads + years_since_first_sample + body_region + chronicity_group + culture_positive,
  data = qc,
  method = "bray",
  by = "margin",
  permutations = 1999,
  strata = qc$patient_id
)

permanova_table <- permanova |>
  as.data.frame() |>
  tibble::rownames_to_column("term") |>
  as_tibble() |>
  rename(
    df = Df,
    sum_of_squares = SumOfSqs,
    r2 = R2,
    f_statistic = F,
    pvalue = `Pr(>F)`
  ) |>
  mutate(
    qvalue = NA_real_,
    term_label = case_when(
      term == "batch_id" ~ "Batch date",
      term == "host_removed_fraction" ~ "Host fraction",
      term == "log10_bacterial_reads" ~ "Bacterial read depth",
      term == "years_since_first_sample" ~ "Years since first patient sample",
      term == "body_region" ~ "Body region",
      term == "chronicity_group" ~ "Chronicity",
      term == "culture_positive" ~ "Culture positivity",
      TRUE ~ term
    )
  )

tested <- permanova_table |> filter(!term %in% c("Residual", "Total"))
permanova_table$qvalue[match(tested$term, permanova_table$term)] <- p.adjust(tested$pvalue, method = "BH")

write_tsv(permanova_table, table_file(29, "adjusted_community_permanova"))
print(permanova_table)


# %% [markdown]
# ## Build Pairwise Distances And Fit Cross-Classified Mixed Models
#
# Two pairwise models are fit:
# one using shared body region and one using exact shared cleaned location.
# This separates broad site similarity from exact-site recurrence.
#

# %%
distance_matrix <- as.matrix(vegdist(rel_abundance, method = "bray"))
pair_rows <- vector("list", length = 0)

for (i in seq_len(nrow(qc) - 1)) {
  for (j in seq((i + 1), nrow(qc))) {
    pair_rows[[length(pair_rows) + 1]] <- tibble(
      sample_a = qc$sample_id[i],
      sample_b = qc$sample_id[j],
      distance = distance_matrix[i, j],
      same_patient = qc$patient_id[i] == qc$patient_id[j],
      same_batch = qc$batch_id[i] == qc$batch_id[j],
      same_body_region = qc$body_region[i] == qc$body_region[j],
      same_location = qc$location[i] == qc$location[j],
      same_chronicity = qc$chronicity_group[i] == qc$chronicity_group[j],
      same_culture_positive = qc$culture_positive[i] == qc$culture_positive[j],
      delta_years_since_first_sample = abs(qc$years_since_first_sample[i] - qc$years_since_first_sample[j]),
      mean_host_fraction = mean(c(qc$host_removed_fraction[i], qc$host_removed_fraction[j])),
      mean_log10_bacterial_reads = mean(c(qc$log10_bacterial_reads[i], qc$log10_bacterial_reads[j]))
    )
  }
}

pair_df <- bind_rows(pair_rows) |>
  mutate(
    sample_a = factor(sample_a),
    sample_b = factor(sample_b),
    same_patient = as.numeric(same_patient),
    same_batch = as.numeric(same_batch),
    same_body_region = as.numeric(same_body_region),
    same_location = as.numeric(same_location),
    same_chronicity = as.numeric(same_chronicity),
    same_culture_positive = as.numeric(same_culture_positive)
  )

pair_model_body_region <- lmer(
  distance ~ same_patient + same_batch + same_body_region + same_chronicity + same_culture_positive + delta_years_since_first_sample + mean_host_fraction + mean_log10_bacterial_reads + (1 | sample_a) + (1 | sample_b),
  data = pair_df,
  REML = FALSE,
  control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
)

pair_model_exact_location <- lmer(
  distance ~ same_patient + same_batch + same_location + same_chronicity + same_culture_positive + delta_years_since_first_sample + mean_host_fraction + mean_log10_bacterial_reads + (1 | sample_a) + (1 | sample_b),
  data = pair_df,
  REML = FALSE,
  control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
)

extract_pairwise_effects <- function(model, model_name) {
  tidy(model, effects = "fixed", conf.int = TRUE) |>
    filter(term != "(Intercept)") |>
    mutate(
      model = model_name,
      term_label = case_when(
        term == "same_patient" ~ "Same patient",
        term == "same_batch" ~ "Same batch date",
        term == "same_body_region" ~ "Same body region",
        term == "same_location" ~ "Same exact location",
        term == "same_chronicity" ~ "Same chronicity",
        term == "same_culture_positive" ~ "Same culture positivity",
        term == "delta_years_since_first_sample" ~ "Elapsed-time gap (years)",
        term == "mean_host_fraction" ~ "Mean host fraction",
        term == "mean_log10_bacterial_reads" ~ "Mean bacterial read depth",
        TRUE ~ term
      )
    )
}

pairwise_effects_all <- bind_rows(
  extract_pairwise_effects(pair_model_body_region, "body_region_model"),
  extract_pairwise_effects(pair_model_exact_location, "exact_location_model")
) |>
  group_by(model) |>
  mutate(qvalue = p.adjust(p.value, method = "BH")) |>
  ungroup()

pairwise_status_all <- tibble(
  model = c("body_region_model", "exact_location_model"),
  n_pairs = nrow(pair_df),
  aic = c(AIC(pair_model_body_region), AIC(pair_model_exact_location)),
  bic = c(BIC(pair_model_body_region), BIC(pair_model_exact_location)),
  logLik = c(as.numeric(logLik(pair_model_body_region)), as.numeric(logLik(pair_model_exact_location))),
  singular = c(isSingular(pair_model_body_region), isSingular(pair_model_exact_location))
)

pairwise_effects <- pairwise_effects_all |> filter(model == "body_region_model")
pairwise_effects_exact <- pairwise_effects_all |> filter(model == "exact_location_model")
pairwise_status <- pairwise_status_all |> filter(model == "body_region_model")
pairwise_status_exact <- pairwise_status_all |> filter(model == "exact_location_model")

write_tsv(pairwise_effects, table_file(30, "pairwise_similarity_mixed_effects"))
write_tsv(pairwise_status, table_file(31, "pairwise_similarity_model_status"))
write_tsv(pairwise_effects_exact, table_file(39, "pairwise_similarity_mixed_effects_exact_location_sensitivity"))
write_tsv(pairwise_status_exact, table_file(40, "pairwise_similarity_model_status_exact_location_sensitivity"))

print(pairwise_effects)
print(pairwise_status)


# %% [markdown]
# ## Estimate Covariate-Adjusted Similarity Margins
#
# To visualize adjusted similarity directly, we estimate marginal Bray-Curtis distances from the fitted pairwise mixed models.
# Other pairwise matching indicators are held at 0, and continuous technical covariates are fixed at their sample means.
#

# %%
margin_specs <- tribble(
  ~model_key, ~focal_term, ~term_label, ~term_type, ~low_label, ~high_label,
  "body_region_model", "same_patient", "Same patient", "binary", "Not shared", "Shared",
  "body_region_model", "same_chronicity", "Same chronicity", "binary", "Not shared", "Shared",
  "body_region_model", "same_body_region", "Same body region", "binary", "Not shared", "Shared",
  "body_region_model", "same_batch", "Same batch date", "binary", "Not shared", "Shared",
  "body_region_model", "same_culture_positive", "Same culture positivity", "binary", "Not shared", "Shared",
  "body_region_model", "delta_years_since_first_sample", "Elapsed-time gap (years)", "continuous", "Shorter gap", "Longer gap",
  "body_region_model", "mean_host_fraction", "Mean host fraction", "continuous", "Lower", "Higher",
  "body_region_model", "mean_log10_bacterial_reads", "Mean bacterial read depth", "continuous", "Lower", "Higher",
  "exact_location_model", "same_location", "Same exact location", "binary", "Not shared", "Shared"
)

model_lookup <- list(
  body_region_model = pair_model_body_region,
  exact_location_model = pair_model_exact_location
)

make_margin_table <- function(model, focal_term, term_label, model_key, term_type, low_label, high_label) {
  at_list <- list(
    same_patient = 0,
    same_batch = 0,
    same_body_region = 0,
    same_location = 0,
    same_chronicity = 0,
    same_culture_positive = 0,
    delta_years_since_first_sample = mean(pair_df$delta_years_since_first_sample),
    mean_host_fraction = mean(pair_df$mean_host_fraction),
    mean_log10_bacterial_reads = mean(pair_df$mean_log10_bacterial_reads)
  )
  focal_values <- if (term_type == "binary") {
    c(0, 1)
  } else {
    as.numeric(quantile(pair_df[[focal_term]], probs = c(0.25, 0.75), na.rm = TRUE))
  }
  at_list[[focal_term]] <- focal_values
  em <- emmeans(model, specs = as.formula(paste("~", focal_term)), at = at_list)
  out <- as_tibble(summary(em, infer = c(TRUE, TRUE))) |>
    arrange(.data[[focal_term]])
  out$focal_level_value <- out[[focal_term]]
  coeff_row <- pairwise_effects_all |>
    filter(model == model_key, term == focal_term) |>
    slice_head(n = 1)
  coeff_pvalue <- if (nrow(coeff_row) > 0) coeff_row$p.value[[1]] else NA_real_
  coeff_qvalue <- if (nrow(coeff_row) > 0) coeff_row$qvalue[[1]] else NA_real_
  out |>
    mutate(
      level = c(low_label, high_label),
      focal_value = focal_level_value,
      comparison_basis = if_else(term_type == "binary", "0_vs_1", "q25_vs_q75"),
      pvalue = coeff_pvalue,
      qvalue = coeff_qvalue,
      term_type = term_type
    ) |>
    transmute(
      model = model_key,
      focal_term = focal_term,
      term_label = term_label,
      term_type = term_type,
      level = level,
      focal_value = focal_value,
      comparison_basis = comparison_basis,
      emmean = emmean,
      std.error = SE,
      conf.low = lower.CL,
      conf.high = upper.CL,
      pvalue = pvalue,
      qvalue = qvalue,
      held_same_batch = 0,
      held_same_culture_positive = 0,
      held_elapsed_time_gap = mean(pair_df$delta_years_since_first_sample),
      held_mean_host_fraction = mean(pair_df$mean_host_fraction),
      held_mean_log10_bacterial_reads = mean(pair_df$mean_log10_bacterial_reads)
    )
}

adjusted_margins_all <- bind_rows(lapply(seq_len(nrow(margin_specs)), function(i) {
  make_margin_table(
    model_lookup[[margin_specs$model_key[[i]]]],
    margin_specs$focal_term[[i]],
    margin_specs$term_label[[i]],
    margin_specs$model_key[[i]],
    margin_specs$term_type[[i]],
    margin_specs$low_label[[i]],
    margin_specs$high_label[[i]]
  )
}))

adjusted_margins <- adjusted_margins_all |> filter(model == "body_region_model")
adjusted_margins_exact <- adjusted_margins_all |> filter(model == "exact_location_model")
write_tsv(adjusted_margins, table_file(38, "pairwise_adjusted_margins"))
write_tsv(adjusted_margins_exact, table_file(41, "pairwise_adjusted_margins_exact_location_sensitivity"))
print(adjusted_margins)


# %% [markdown]
# ## Review Numbered Table Outputs
#

# %%
outputs <- tibble(
  output = c(
    table_file(29, "adjusted_community_permanova"),
    table_file(30, "pairwise_similarity_mixed_effects"),
    table_file(31, "pairwise_similarity_model_status"),
    table_file(38, "pairwise_adjusted_margins"),
    table_file(39, "pairwise_similarity_mixed_effects_exact_location_sensitivity"),
    table_file(40, "pairwise_similarity_model_status_exact_location_sensitivity"),
    table_file(41, "pairwise_adjusted_margins_exact_location_sensitivity")
  )
)
print(outputs)
