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
# Minimal model set for README-linked outputs:
# - pairwise mixed-effects fixed-effect table
# - adjusted margins table
#

# %%
options(width = 140)
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(vegan)
  library(lme4)
  library(lmerTest)
  library(broom.mixed)
  library(emmeans)
})

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

table_file <- function(number, slug) {
  prefix <- table_id_map[[as.character(number)]]
  file.path(table_dir, sprintf("table_%s_%s.tsv", prefix, slug))
}


# %% [markdown]
# ## Load QC-Passing Data
#

# %%
qc <- read_tsv(table_file(2, "qc_metrics"), show_col_types = FALSE) |>
  mutate(
    model_qc_pass = tolower(as.character(model_qc_pass)) %in% c("true", "t", "1"),
    culture_date = as.Date(culture_date),
    batch_id = factor(batch_id),
    culture_positive = factor(
      if_else(as.logical(culture_positive), "positive", "negative"),
      levels = c("negative", "positive")
    ),
    patient_id = factor(sprintf("%02d", as.integer(patient_id))),
    body_region = factor(
      body_region,
      levels = c("lower_extremity", "head_neck", "upper_extremity", "trunk_perineum", "others")
    ),
    chronicity_group = factor(
      chronicity_group,
      levels = c("unknown", "acute_like", "chronic_like", "mixed")
    )
  ) |>
  filter(model_qc_pass) |>
  select(
    sample_id,
    patient_id,
    batch_id,
    years_since_first_sample,
    body_region,
    chronicity_group,
    culture_positive,
    host_removed_fraction,
    log10_bacterial_reads
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


# %% [markdown]
# ## Pairwise Mixed-Effects Model (Body Region)
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
    same_chronicity = as.numeric(same_chronicity),
    same_culture_positive = as.numeric(same_culture_positive)
  )

pair_model <- lmer(
  distance ~ same_patient + same_batch + same_body_region + same_chronicity + same_culture_positive + delta_years_since_first_sample + mean_host_fraction + mean_log10_bacterial_reads + (1 | sample_a) + (1 | sample_b),
  data = pair_df,
  REML = FALSE,
  control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
)

pairwise_effects <- tidy(pair_model, effects = "fixed", conf.int = TRUE) |>
  filter(term != "(Intercept)") |>
  mutate(
    model = "body_region_model",
    term_label = case_when(
      term == "same_patient" ~ "Same patient",
      term == "same_batch" ~ "Same batch date",
      term == "same_body_region" ~ "Same body region",
      term == "same_chronicity" ~ "Same chronicity",
      term == "same_culture_positive" ~ "Same culture positivity",
      term == "delta_years_since_first_sample" ~ "Elapsed-time gap (years)",
      term == "mean_host_fraction" ~ "Mean host fraction",
      term == "mean_log10_bacterial_reads" ~ "Mean bacterial read depth",
      TRUE ~ term
    ),
    qvalue = p.adjust(p.value, method = "BH")
  )

write_tsv(pairwise_effects, table_file(30, "pairwise_similarity_mixed_effects"))
print(pairwise_effects)


# %% [markdown]
# ## Adjusted Margins For Shared vs Non-Shared States
#

# %%
margin_specs <- tribble(
  ~focal_term, ~term_label,
  "same_patient", "Same patient",
  "same_chronicity", "Same chronicity",
  "same_body_region", "Same body region"
)

make_margin_table <- function(focal_term, term_label) {
  at_list <- list(
    same_patient = 0,
    same_batch = 0,
    same_body_region = 0,
    same_chronicity = 0,
    same_culture_positive = 0,
    delta_years_since_first_sample = mean(pair_df$delta_years_since_first_sample),
    mean_host_fraction = mean(pair_df$mean_host_fraction),
    mean_log10_bacterial_reads = mean(pair_df$mean_log10_bacterial_reads)
  )
  at_list[[focal_term]] <- c(0, 1)
  em <- emmeans(pair_model, specs = as.formula(paste("~", focal_term)), at = at_list)
  out <- as_tibble(summary(em, infer = c(TRUE, TRUE)))
  out$focal_level_value <- out[[focal_term]]
  out |>
    transmute(
      model = "body_region_model",
      focal_term = focal_term,
      term_label = term_label,
      level = if_else(focal_level_value == 1, "Shared", "Not shared"),
      emmean = emmean,
      std.error = SE,
      conf.low = lower.CL,
      conf.high = upper.CL,
      held_same_batch = 0,
      held_same_culture_positive = 0,
      held_elapsed_time_gap = mean(pair_df$delta_years_since_first_sample),
      held_mean_host_fraction = mean(pair_df$mean_host_fraction),
      held_mean_log10_bacterial_reads = mean(pair_df$mean_log10_bacterial_reads)
    )
}

adjusted_margins <- bind_rows(lapply(seq_len(nrow(margin_specs)), function(i) {
  make_margin_table(margin_specs$focal_term[[i]], margin_specs$term_label[[i]])
}))

write_tsv(adjusted_margins, table_file(38, "pairwise_adjusted_margins"))
print(adjusted_margins)


# %% [markdown]
# ## Review Numbered Table Outputs
#

# %%
outputs <- tibble(
  output = c(
    table_file(30, "pairwise_similarity_mixed_effects"),
    table_file(38, "pairwise_adjusted_margins")
  )
)
print(outputs)
