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
# # 11. Host Fraction Beta-Binomial Mixed Model
#
# This notebook upgrades the earlier host-fraction regression to a count-based mixed model.
# The response is modeled as Bracken human species reads out of Bracken root reads
# from the pre-host-filter reports,
# so library-size scaling is handled through the binomial denominator rather than added again as a separate nuisance covariate.
# Absolute culture date is treated as technical batch, while patient-relative elapsed time is treated as biology.
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
                      library(glmmTMB)
                      library(broom.mixed)
                    })


# %% [markdown]
# ## Load QC Data And Define The Host Model Inputs
#
# The model keeps a limited fixed-effect set to avoid overadjustment:
# body site, chronicity, broad culture positivity, and patient-relative elapsed time.
# Patient and culture-date batch are handled as random intercepts.
#

# %%
qc <- read_tsv(table_file(2, "qc_metrics"), show_col_types = FALSE) |>
  mutate(
    culture_date = as.Date(culture_date),
    human_reads = pmax(human_species_reads, 0),
    non_human_reads = pmax(bracken_total_reads - human_species_reads, 0),
    culture_positive = factor(if_else(as.logical(culture_positive), "positive", "negative"),
                              levels = c("negative", "positive")),
    patient_id = factor(sprintf("%02d", as.integer(patient_id))),
    batch_id = factor(batch_id),
    body_region = factor(body_region, levels = c("lower_extremity", "head_neck", "upper_extremity", "trunk_perineum", "unknown")),
    chronicity_group = factor(chronicity_group, levels = c("unknown", "acute_like", "chronic_like", "mixed")),
    upper_extremity_binary = factor(if_else(body_region == "upper_extremity", "upper_extremity", "other"),
                                    levels = c("other", "upper_extremity")),
    acute_like_binary = factor(if_else(chronicity_group == "acute_like", "acute_like", "other"),
                               levels = c("other", "acute_like"))
  ) |>
  filter(
    !is.na(human_reads),
    !is.na(non_human_reads),
    bracken_total_reads > 0,
    !is.na(body_region),
    !is.na(chronicity_group),
    !is.na(culture_positive),
    !is.na(years_since_first_sample),
    !is.na(batch_id)
  )

host_model_formula <- "cbind(human_reads, non_human_reads) ~ body_region + chronicity_group + culture_positive + years_since_first_sample + (1 | patient_id) + (1 | batch_id)"

host_input_summary <- tibble(
  n_samples = nrow(qc),
  n_patients = n_distinct(qc$patient_id),
  n_batches = n_distinct(qc$batch_id),
  response = "cbind(human_reads, non_human_reads)",
  fixed_effects = "body_region + chronicity_group + culture_positive + years_since_first_sample",
  random_effects = "(1 | patient_id) + (1 | batch_id)",
  family = "betabinomial(link = 'logit')"
)

print(host_input_summary)


# %% [markdown]
# ## Fit The Beta-Binomial Mixed Model
#

# %%
extract_re_var <- function(fit, group_name) {
  vc <- VarCorr(fit)$cond[[group_name]]
  if (is.null(vc)) {
    return(NA_real_)
  }
  stddev <- attr(vc, "stddev")
  if (is.null(stddev) || length(stddev) == 0) {
    return(NA_real_)
  }
  as.numeric(stddev[1]^2)
}

fit_host_candidate <- function(model_name, formula_text) {
  fit <- tryCatch(
    glmmTMB(
      formula = as.formula(formula_text),
      data = qc,
      family = betabinomial(link = "logit"),
      control = glmmTMBControl(optCtrl = list(iter.max = 1e4, eval.max = 1e4))
    ),
    error = function(e) e
  )
  if (inherits(fit, "error")) {
    return(list(
      fit = NULL,
      status = tibble(
        model = model_name,
        formula = formula_text,
        status = "failed",
        n_samples = nrow(qc),
        aic = NA_real_,
        bic = NA_real_,
        logLik = NA_real_,
        patient_var = NA_real_,
        batch_var = NA_real_,
        pd_hessian = FALSE,
        warning = as.character(fit$message)
      )
    ))
  }

  status <- tibble(
    model = model_name,
    formula = formula_text,
    status = "ok",
    n_samples = nrow(qc),
    aic = AIC(fit),
    bic = BIC(fit),
    logLik = as.numeric(logLik(fit)),
    patient_var = extract_re_var(fit, "patient_id"),
    batch_var = extract_re_var(fit, "batch_id"),
    pd_hessian = isTRUE(fit$sdr$pdHess),
    warning = if_else(isTRUE(fit$sdr$pdHess), "", "non-positive-definite Hessian or optimizer warning")
  )
  list(fit = fit, status = status)
}

full_fixed_formula <- "cbind(human_reads, non_human_reads) ~ body_region + chronicity_group + culture_positive + years_since_first_sample"
full_random_formula <- paste(full_fixed_formula, "+ (1 | patient_id) + (1 | batch_id)")
patient_only_formula <- paste(full_fixed_formula, "+ (1 | patient_id)")
batch_only_formula <- paste(full_fixed_formula, "+ (1 | batch_id)")
no_body_region_formula <- "cbind(human_reads, non_human_reads) ~ chronicity_group + culture_positive + years_since_first_sample + (1 | patient_id) + (1 | batch_id)"
no_chronicity_formula <- "cbind(human_reads, non_human_reads) ~ body_region + culture_positive + years_since_first_sample + (1 | patient_id) + (1 | batch_id)"

candidate_results <- list(
  fit_host_candidate("fixed_only", full_fixed_formula),
  fit_host_candidate("patient_plus_batch", full_random_formula),
  fit_host_candidate("batch_only", batch_only_formula),
  fit_host_candidate("patient_only", patient_only_formula)
)

host_status <- bind_rows(lapply(candidate_results, `[[`, "status"))
reported_model_name <- "patient_plus_batch"
if (!reported_model_name %in% host_status$model || host_status$status[match(reported_model_name, host_status$model)] != "ok") {
  ok_status <- host_status |>
    filter(status == "ok")
  preferred_status <- ok_status |>
    filter(pd_hessian) |>
    arrange(aic)
  if (nrow(preferred_status) == 0) {
    preferred_status <- ok_status |> arrange(aic)
  }
  reported_model_name <- preferred_status$model[1]
}
host_model <- candidate_results[[match(reported_model_name, host_status$model)]]$fit

reduced_results <- list(
  fit_host_candidate("no_body_region", no_body_region_formula),
  fit_host_candidate("no_chronicity", no_chronicity_formula)
)

contrast_specs <- tribble(
  ~model_name, ~formula_text, ~target_term, ~term_label,
  "upper_extremity_contrast",
  "cbind(human_reads, non_human_reads) ~ upper_extremity_binary + chronicity_group + culture_positive + years_since_first_sample + (1 | patient_id) + (1 | batch_id)",
  "upper_extremity_binaryupper_extremity",
  "Planned contrast: upper extremity vs all other body regions",
  "acute_like_contrast",
  "cbind(human_reads, non_human_reads) ~ body_region + acute_like_binary + culture_positive + years_since_first_sample + (1 | patient_id) + (1 | batch_id)",
  "acute_like_binaryacute_like",
  "Planned contrast: acute-like vs all other chronicity groups"
)

extract_loglik_df <- function(fit) {
  ll <- logLik(fit)
  list(logLik = as.numeric(ll), df = attr(ll, "df"))
}

get_fit_from_results <- function(results, model_name) {
  model_names <- vapply(results, function(x) x$status$model[[1]], character(1))
  idx <- which(model_names == model_name)
  if (length(idx) == 0) {
    return(NULL)
  }
  results[[idx[1]]]$fit
}

compare_models <- function(full_fit, reduced_fit, effect_label, full_name, reduced_name, record_type = "random_effect_test") {
  if (is.null(full_fit) || is.null(reduced_fit)) {
    return(tibble(
      model = paste0(full_name, "_vs_", reduced_name),
      formula = NA_character_,
      status = "comparison_failed",
      n_samples = nrow(qc),
      aic = NA_real_,
      bic = NA_real_,
      logLik = NA_real_,
      patient_var = NA_real_,
      batch_var = NA_real_,
      pd_hessian = NA,
      warning = paste("Comparison failed for", effect_label),
      record_type = record_type,
      tested_effect = effect_label,
      full_model = full_name,
      reduced_model = reduced_name,
      lrt_statistic = NA_real_,
      df_diff = NA_real_,
      pvalue_chisq = NA_real_,
      pvalue_boundary = NA_real_
    ))
  }
  full_info <- extract_loglik_df(full_fit)
  reduced_info <- extract_loglik_df(reduced_fit)
  lrt_stat <- max(0, 2 * (full_info$logLik - reduced_info$logLik))
  df_diff <- full_info$df - reduced_info$df
  p_chisq <- pchisq(lrt_stat, df = df_diff, lower.tail = FALSE)
  p_boundary <- ifelse(df_diff == 1, 0.5 * p_chisq, NA_real_)
  tibble(
    model = paste0(full_name, "_vs_", reduced_name),
    formula = NA_character_,
    status = "ok",
    n_samples = nrow(qc),
    aic = NA_real_,
    bic = NA_real_,
    logLik = NA_real_,
    patient_var = NA_real_,
    batch_var = NA_real_,
    pd_hessian = NA,
    warning = "",
    record_type = record_type,
    tested_effect = effect_label,
    full_model = full_name,
    reduced_model = reduced_name,
    lrt_statistic = lrt_stat,
    df_diff = df_diff,
    pvalue_chisq = p_chisq,
    pvalue_boundary = p_boundary
  )
}

fixed_only_fit <- get_fit_from_results(candidate_results, "fixed_only")
patient_fit <- get_fit_from_results(candidate_results, "patient_only")
batch_fit <- get_fit_from_results(candidate_results, "batch_only")
patient_plus_batch_fit <- get_fit_from_results(candidate_results, "patient_plus_batch")
no_body_region_fit <- get_fit_from_results(reduced_results, "no_body_region")
no_chronicity_fit <- get_fit_from_results(reduced_results, "no_chronicity")

re_tests <- bind_rows(
  compare_models(patient_fit, fixed_only_fit, "patient_only_vs_fixed", "patient_only", "fixed_only"),
  compare_models(batch_fit, fixed_only_fit, "batch_only_vs_fixed", "batch_only", "fixed_only"),
  compare_models(patient_plus_batch_fit, patient_fit, "batch_added_to_patient", "patient_plus_batch", "patient_only"),
  compare_models(patient_plus_batch_fit, batch_fit, "patient_added_to_batch", "patient_plus_batch", "batch_only")
)

fixed_effect_tests <- bind_rows(
  compare_models(patient_plus_batch_fit, no_body_region_fit, "body_region_overall", "patient_plus_batch", "no_body_region", "fixed_effect_test"),
  compare_models(patient_plus_batch_fit, no_chronicity_fit, "chronicity_group_overall", "patient_plus_batch", "no_chronicity", "fixed_effect_test")
)

fit_contrast_model <- function(model_name, formula_text, target_term, term_label) {
  result <- fit_host_candidate(model_name, formula_text)
  status <- result$status |>
    mutate(
      record_type = "contrast_model_fit",
      tested_effect = NA_character_,
      full_model = NA_character_,
      reduced_model = NA_character_,
      lrt_statistic = NA_real_,
      df_diff = NA_real_,
      pvalue_chisq = NA_real_,
      pvalue_boundary = NA_real_
    )
  effect <- tibble(
    model_name = model_name,
    analysis_type = "targeted_contrast",
    adjustment_family = "planned_contrast_terms",
    term = target_term,
    term_label = term_label,
    estimate = NA_real_,
    std.error = NA_real_,
    conf.low = NA_real_,
    conf.high = NA_real_,
    odds_ratio = NA_real_,
    conf.low.or = NA_real_,
    conf.high.or = NA_real_,
    p.value = NA_real_
  )
  if (!is.null(result$fit)) {
    tidy_fit <- tidy(result$fit, effects = "fixed", component = "cond")
    target_row <- tidy_fit |> filter(term == target_term)
    if (nrow(target_row) == 1) {
      effect <- target_row |>
        transmute(
          model_name = model_name,
          analysis_type = "targeted_contrast",
          adjustment_family = "planned_contrast_terms",
          term = term,
          term_label = term_label,
          estimate = estimate,
          std.error = std.error,
          conf.low = estimate - 1.96 * std.error,
          conf.high = estimate + 1.96 * std.error,
          odds_ratio = exp(estimate),
          conf.low.or = exp(conf.low),
          conf.high.or = exp(conf.high),
          p.value = p.value
        )
    }
  }
  list(status = status, effect = effect)
}

contrast_results <- lapply(seq_len(nrow(contrast_specs)), function(i) {
  fit_contrast_model(
    contrast_specs$model_name[[i]],
    contrast_specs$formula_text[[i]],
    contrast_specs$target_term[[i]],
    contrast_specs$term_label[[i]]
  )
})

contrast_status <- bind_rows(lapply(contrast_results, `[[`, "status"))
contrast_effects <- bind_rows(lapply(contrast_results, `[[`, "effect")) |>
  mutate(
    qvalue = p.adjust(p.value, method = "BH"),
    posthoc_family = "planned_contrast_terms",
    posthoc_qvalue = qvalue
  )

host_status <- host_status |>
  mutate(
    record_type = "model_fit",
    tested_effect = NA_character_,
    full_model = NA_character_,
    reduced_model = NA_character_,
    lrt_statistic = NA_real_,
    df_diff = NA_real_,
    pvalue_chisq = NA_real_,
    pvalue_boundary = NA_real_
  ) |>
  bind_rows(
    bind_rows(lapply(reduced_results, `[[`, "status")) |>
      mutate(
        record_type = "fixed_effect_reduced_model_fit",
        tested_effect = NA_character_,
        full_model = NA_character_,
        reduced_model = NA_character_,
        lrt_statistic = NA_real_,
        df_diff = NA_real_,
        pvalue_chisq = NA_real_,
        pvalue_boundary = NA_real_
      ),
    contrast_status,
    re_tests,
    fixed_effect_tests
  )

host_effects <- tidy(host_model, effects = "fixed", component = "cond") |>
  filter(term != "(Intercept)") |>
  mutate(
    model_name = reported_model_name,
    analysis_type = "full_model",
    adjustment_family = "full_model_terms",
    conf.low = estimate - 1.96 * std.error,
    conf.high = estimate + 1.96 * std.error,
    odds_ratio = exp(estimate),
    conf.low.or = exp(conf.low),
    conf.high.or = exp(conf.high),
    qvalue = p.adjust(p.value, method = "BH"),
    selected_model = reported_model_name,
    posthoc_family = case_when(
      str_detect(term, "^body_region") ~ "body_region_terms",
      str_detect(term, "^chronicity_group") ~ "chronicity_terms",
      term == "culture_positivepositive" ~ "culture_positive_term",
      term == "years_since_first_sample" ~ "elapsed_time_term",
      TRUE ~ "other_terms"
    ),
    term_label = case_when(
      term == "years_since_first_sample" ~ "Per year since first patient sample",
      term == "culture_positivepositive" ~ "Culture positive: yes",
      term == "body_regionhead_neck" ~ "Body site: head / neck",
      term == "body_regionupper_extremity" ~ "Body site: upper extremity",
      term == "body_regiontrunk_perineum" ~ "Body site: trunk / perineum",
      term == "body_regionunknown" ~ "Body site: unknown",
      term == "chronicity_groupacute_like" ~ "Chronicity: acute-like",
      term == "chronicity_groupchronic_like" ~ "Chronicity: chronic-like",
      term == "chronicity_groupmixed" ~ "Chronicity: mixed",
      TRUE ~ term
    )
  ) |>
  group_by(posthoc_family) |>
  mutate(posthoc_qvalue = p.adjust(p.value, method = "BH")) |>
  ungroup() |>
  arrange(qvalue, p.value)

host_effects <- bind_rows(
  host_effects,
  contrast_effects
) |>
  arrange(adjustment_family, qvalue, p.value)

write_tsv(host_effects, table_file(27, "host_beta_binomial_effects"))
write_tsv(host_status, table_file(28, "host_beta_binomial_status"))

print(host_effects)
print(host_status)


# %% [markdown]
# ## Summarize Positive And Negative Results
#

# %%
plot_df <- host_effects |>
  filter(!is.na(odds_ratio)) |>
  mutate(
    panel = if_else(analysis_type == "full_model", "Full patient-plus-batch model", "Planned 1-df contrast models"),
    term_label = factor(term_label, levels = rev(unique(term_label))),
    display_qvalue = if_else(is.na(posthoc_qvalue), qvalue, posthoc_qvalue),
    significant = display_qvalue <= 0.1
  )

figure_11 <- ggplot(plot_df, aes(x = odds_ratio, y = term_label, color = significant)) +
  geom_vline(xintercept = 1, linewidth = 0.5, linetype = "dashed", color = "grey50") +
  geom_errorbarh(aes(xmin = conf.low.or, xmax = conf.high.or), height = 0.18, linewidth = 0.7) +
  geom_point(size = 2.6) +
  scale_x_log10() +
  scale_color_manual(values = c("TRUE" = "#b22222", "FALSE" = "#3b6a8f")) +
  facet_grid(panel ~ ., scales = "free_y", space = "free_y") +
  labs(
    title = "Host fraction beta-binomial model and planned host contrasts",
    x = "Odds ratio for host fraction",
    y = NULL,
    color = "Targeted q <= 0.1"
  ) +
  theme(legend.position = "top")

ggsave(
  figure_file(11, "host_beta_binomial"),
  figure_11,
  width = 11,
  height = 7.5,
  device = grDevices::svg
)
print(figure_11)

body_region_test <- host_status |>
  filter(record_type == "fixed_effect_test", tested_effect == "body_region_overall") |>
  slice(1)
chronicity_test <- host_status |>
  filter(record_type == "fixed_effect_test", tested_effect == "chronicity_group_overall") |>
  slice(1)
planned_hits <- host_effects |>
  filter(analysis_type == "targeted_contrast") |>
  arrange(qvalue, p.value)
posthoc_hits <- host_effects |>
  filter(analysis_type == "full_model", posthoc_family %in% c("body_region_terms", "chronicity_terms")) |>
  arrange(posthoc_qvalue, p.value)

host_findings <- tibble(
  finding = c(
    sprintf("Positive result: %d host-model terms reached q <= 0.1.", sum(host_effects$qvalue <= 0.1, na.rm = TRUE)),
    sprintf("Positive result: the reported host model now conditions on body site, chronicity, culture positivity, and patient-relative elapsed time at the same time; the reported random-effect structure was %s.", reported_model_name),
    sprintf("Positive result: omnibus fixed-effect tests gave p=%.3g for body_region and p=%.3g for chronicity_group in the full patient-plus-batch model.", body_region_test$pvalue_chisq[[1]], chronicity_test$pvalue_chisq[[1]]),
    sprintf("Positive result: factor-specific post hoc BH gave q=%.3g for %s.", posthoc_hits$posthoc_qvalue[[1]], posthoc_hits$term_label[[1]]),
    sprintf("Positive result: the strongest planned contrast was %s with OR %.2f and q=%.3g.", planned_hits$term_label[[1]], planned_hits$odds_ratio[[1]], planned_hits$qvalue[[1]]),
    "Positive result: random-effect contribution is now summarized with explicit likelihood-ratio comparisons against reduced models.",
    "Negative result: total sequencing depth is not added as a separate covariate here because the beta-binomial denominator already uses trimmed read pairs.",
    "Negative result: absolute date is not interpreted biologically here; it is absorbed as a technical batch random effect instead.",
    "Negative result: this still does not include every culture subgroup simultaneously because that would overparameterize the host model for this cohort."
  )
)

print(host_findings)

