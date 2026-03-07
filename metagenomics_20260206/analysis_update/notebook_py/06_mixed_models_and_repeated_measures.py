# ---
# jupyter:
#   jupytext:
#     formats: ipynb,notebook_py//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python (eb)
#     language: python
#     name: eb
# ---

# %% [markdown]
# # 06. Mixed Models And Repeated Measures
#
# This notebook revisits the main host and taxon associations with patient and culture-date batch variance components.
# It also records the single-group versus patient+batch diagnostics that motivated the updated specification.
#

# %%
from pathlib import Path
import sys

import pandas as pd
from IPython.display import Markdown, SVG, display

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import workflow_core as wc

context, base_data, base, advanced = wc.bootstrap_notebook()


# %% [markdown]
# ## Define The Patient-And-Batch Diagnostic Fits
#

# %%
import shutil
import warnings

import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2


def _best_mixed_fit(candidates):
    converged = [
        item for item in candidates if bool(getattr(item[0], "converged", False))
    ]
    pool = converged if converged else candidates
    return min(pool, key=lambda item: item[0].aic)


def fit_single_group_model(data, formula, outcome, structure_name, group_col):
    candidates = []
    last_error = ""
    for method in ["lbfgs", "powell", "bfgs"]:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = smf.mixedlm(
                    formula=formula,
                    data=data.copy(),
                    groups=data[group_col],
                ).fit(reml=False, method=method, maxiter=500, disp=False)
            candidates.append((fit, [str(item.message) for item in caught], method))
        except Exception as exc:
            last_error = repr(exc)
    if not candidates:
        return {
            "outcome": outcome,
            "structure": structure_name,
            "status": "failed",
            "converged": False,
            "aic": np.nan,
            "optimizer": "",
            "patient_var": np.nan,
            "batch_var": np.nan,
            "warning_count": 0,
            "warnings": "",
            "error": last_error,
        }

    fit, warning_messages, optimizer = _best_mixed_fit(candidates)
    variance_value = float(fit.params.get("Group Var", np.nan))
    return {
        "outcome": outcome,
        "structure": structure_name,
        "status": "ok",
        "converged": bool(getattr(fit, "converged", False)),
        "aic": float(fit.aic),
        "optimizer": optimizer,
        "patient_var": variance_value if structure_name == "patient_only" else np.nan,
        "batch_var": variance_value if structure_name == "batch_only" else np.nan,
        "warning_count": len(warning_messages),
        "warnings": " | ".join(sorted(set(warning_messages)))[:2000],
        "error": "",
    }


def collect_optimizer_attempts(
    data,
    formula,
    outcome,
    structure_name,
    group_col,
    vc_formula=None,
    constant_group=False,
):
    frame = data.copy()
    if constant_group:
        frame[group_col] = "all"

    rows = []
    candidates = []
    for method in ["lbfgs", "powell", "bfgs"]:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = smf.mixedlm(
                    formula=formula,
                    data=frame,
                    groups=frame[group_col],
                    vc_formula=vc_formula,
                ).fit(reml=False, method=method, maxiter=500, disp=False)
            warning_messages = [str(item.message) for item in caught]
            row = {
                "outcome": outcome,
                "model_name": advanced.HOST_MODEL_NAME,
                "structure": structure_name,
                "optimizer": method,
                "status": "ok",
                "converged": bool(getattr(fit, "converged", False)),
                "aic": float(fit.aic),
                "bic": float(fit.bic),
                "llf": float(fit.llf),
                "patient_var": float(fit.params.get("patient Var", np.nan))
                if vc_formula is not None
                else (
                    float(fit.params.get("Group Var", np.nan))
                    if structure_name == "patient_only"
                    else np.nan
                ),
                "batch_var": float(fit.params.get("batch Var", np.nan))
                if vc_formula is not None
                else (
                    float(fit.params.get("Group Var", np.nan))
                    if structure_name == "batch_only"
                    else np.nan
                ),
                "warning_count": len(warning_messages),
                "warnings": " | ".join(sorted(set(warning_messages)))[:2000],
                "error": "",
                "selected": False,
            }
            rows.append(row)
            candidates.append((fit, warning_messages, method))
        except Exception as exc:
            rows.append(
                {
                    "outcome": outcome,
                    "model_name": advanced.HOST_MODEL_NAME,
                    "structure": structure_name,
                    "optimizer": method,
                    "status": "failed",
                    "converged": False,
                    "aic": np.nan,
                    "bic": np.nan,
                    "llf": np.nan,
                    "patient_var": np.nan,
                    "batch_var": np.nan,
                    "warning_count": 0,
                    "warnings": "",
                    "error": repr(exc),
                    "selected": False,
                }
            )

    if candidates:
        selected_fit, _, selected_optimizer = _best_mixed_fit(candidates)
        selected_aic = float(selected_fit.aic)
        for row in rows:
            if (
                row["status"] == "ok"
                and row["optimizer"] == selected_optimizer
                and np.isfinite(row["aic"])
                and abs(float(row["aic"]) - selected_aic) < 1e-8
            ):
                row["selected"] = True
                break
    return rows


def fit_patient_structure_diagnostics(qc):
    host_df = qc.dropna(
        subset=[
            "host_logit",
            "body_region",
            "patient_id",
            "batch_id",
            "chronicity_group",
            "culture_positive_label",
            "years_since_first_sample",
        ]
    ).copy()
    host_formula = advanced.HOST_FORMULAS[advanced.HOST_MODEL_NAME]
    rows = []
    rows.extend(
        collect_optimizer_attempts(
            host_df,
            host_formula,
            "host_logit",
            "patient_only",
            "patient_id",
        )
    )
    rows.extend(
        collect_optimizer_attempts(
            host_df,
            host_formula,
            "host_logit",
            "batch_only",
            "batch_id",
        )
    )
    rows.extend(
        collect_optimizer_attempts(
            host_df,
            host_formula,
            "host_logit",
            "patient_plus_batch",
            "all_group",
            vc_formula={"patient": "0 + C(patient_id)", "batch": "0 + C(batch_id)"},
            constant_group=True,
        )
    )
    return pd.DataFrame(rows)


def fit_patient_plus_batch_model(data, formula):
    frame = data.copy()
    frame["all_group"] = "all"
    vc_formula = {"patient": "0 + C(patient_id)", "batch": "0 + C(batch_id)"}
    candidates = []
    last_error = ""
    for method in ["lbfgs", "powell", "bfgs"]:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = smf.mixedlm(
                    formula=formula,
                    data=frame,
                    groups=frame["all_group"],
                    vc_formula=vc_formula,
                ).fit(reml=False, method=method, maxiter=500, disp=False)
            candidates.append((fit, [str(item.message) for item in caught], method))
        except Exception as exc:
            last_error = repr(exc)
    if not candidates:
        raise RuntimeError(last_error)
    fit, warning_messages, optimizer = _best_mixed_fit(candidates)
    status = {
        "converged": bool(getattr(fit, "converged", False)),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "llf": float(fit.llf),
        "patient_var": float(fit.params.get("patient Var", np.nan)),
        "batch_var": float(fit.params.get("batch Var", np.nan)),
        "optimizer": optimizer,
        "warning_count": len(warning_messages),
        "warnings": " | ".join(sorted(set(warning_messages)))[:2000],
    }
    return fit, status


def fit_best_single_group_model(data, formula, group_col):
    candidates = []
    last_error = ""
    for method in ["lbfgs", "powell", "bfgs"]:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = smf.mixedlm(
                    formula=formula,
                    data=data.copy(),
                    groups=data[group_col],
                ).fit(reml=False, method=method, maxiter=500, disp=False)
            candidates.append((fit, [str(item.message) for item in caught], method))
        except Exception as exc:
            last_error = repr(exc)
    if not candidates:
        raise RuntimeError(last_error)
    fit, warning_messages, optimizer = _best_mixed_fit(candidates)
    status = {
        "converged": bool(getattr(fit, "converged", False)),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "llf": float(fit.llf),
        "group_var": float(fit.params.get("Group Var", np.nan)),
        "optimizer": optimizer,
        "warning_count": len(warning_messages),
        "warnings": " | ".join(sorted(set(warning_messages)))[:2000],
    }
    return fit, status


def build_random_effect_test_row(
    model_name,
    n_samples,
    tested_effect,
    full_model,
    reduced_model,
    full_llf,
    reduced_llf,
):
    lrt_stat = max(0.0, 2.0 * (full_llf - reduced_llf))
    p_chisq = chi2.sf(lrt_stat, 1)
    return {
        "outcome": "host_logit",
        "model_name": model_name,
        "n_samples": n_samples,
        "converged": np.nan,
        "aic": np.nan,
        "bic": np.nan,
        "llf": np.nan,
        "patient_var": np.nan,
        "batch_var": np.nan,
        "optimizer": "",
        "warning_count": np.nan,
        "warnings": "",
        "record_type": "random_effect_test",
        "tested_effect": tested_effect,
        "full_model": full_model,
        "reduced_model": reduced_model,
        "lrt_statistic": lrt_stat,
        "df_diff": 1.0,
        "pvalue_chisq": p_chisq,
        "pvalue_boundary": 0.5 * p_chisq,
    }


def fit_host_random_effect_tests(qc):
    rows = []
    for model_name, formula in advanced.HOST_FORMULAS.items():
        needed = [
            "host_logit",
            "body_region",
            "patient_id",
            "batch_id",
            "culture_positive_label",
            "years_since_first_sample",
        ]
        if "chronicity_group" in formula:
            needed.append("chronicity_group")
        host_df = qc.dropna(subset=needed).copy()

        fixed_fit = smf.ols(formula, data=host_df).fit()
        patient_fit, _ = fit_best_single_group_model(host_df, formula, "patient_id")
        batch_fit, _ = fit_best_single_group_model(host_df, formula, "batch_id")
        full_fit, _ = fit_patient_plus_batch_model(host_df, formula)

        rows.append(
            build_random_effect_test_row(
                model_name,
                host_df.shape[0],
                "patient_only_vs_fixed",
                "patient_only",
                "fixed_only",
                patient_fit.llf,
                fixed_fit.llf,
            )
        )
        rows.append(
            build_random_effect_test_row(
                model_name,
                host_df.shape[0],
                "batch_only_vs_fixed",
                "batch_only",
                "fixed_only",
                batch_fit.llf,
                fixed_fit.llf,
            )
        )
        rows.append(
            build_random_effect_test_row(
                model_name,
                host_df.shape[0],
                "batch_added_to_patient",
                "patient_plus_batch",
                "patient_only",
                full_fit.llf,
                patient_fit.llf,
            )
        )
        rows.append(
            build_random_effect_test_row(
                model_name,
                host_df.shape[0],
                "patient_added_to_batch",
                "patient_plus_batch",
                "batch_only",
                full_fit.llf,
                batch_fit.llf,
            )
        )

    return pd.DataFrame(rows)


# %% [markdown]
# ## Fit The Mixed Models And Save The Numbered Outputs
#

# %%
advanced_context = wc.advanced_analysis_context(context)
advanced.ensure_dirs(advanced_context)

qc = base_data["qc"].copy()
species_bac = base_data["species_bac"]

diagnostics = fit_patient_structure_diagnostics(qc)
host_cluster_fit, host_cluster = base.fit_host_model(qc)
species_cluster = base.fit_species_models(qc, species_bac)

host_effects, host_status = advanced.fit_host_models(qc)
host_status = host_status.assign(
    record_type="model_fit",
    tested_effect=np.nan,
    full_model=np.nan,
    reduced_model=np.nan,
    lrt_statistic=np.nan,
    df_diff=np.nan,
    pvalue_chisq=np.nan,
    pvalue_boundary=np.nan,
)
host_status = pd.concat(
    [host_status, fit_host_random_effect_tests(qc)], ignore_index=True
)
species_effects, species_status = advanced.fit_species_models(qc, species_bac)
comparison = advanced.build_comparison_table(
    host_effects,
    species_effects,
    host_cluster.assign(outcome="host_logit"),
    species_cluster,
)

advanced.make_host_comparison_figure(
    host_effects,
    host_cluster.assign(outcome="host_logit"),
    advanced_context,
)
advanced.make_species_mixed_figure(species_effects, advanced_context)

host_source = context.figure_dir / "advanced_figure_01_host_compare.svg"
host_source_jpg = context.figure_dir / "advanced_figure_01_host_compare.jpg"
species_source = context.figure_dir / "advanced_figure_02_species_mixed.svg"
species_source_jpg = context.figure_dir / "advanced_figure_02_species_mixed.jpg"
host_dest = wc.figure_path(context, 5, "host_model_compare")
host_dest_jpg = host_dest.with_suffix(".jpg")
species_dest = wc.figure_path(context, 6, "species_mixed_effects")
species_dest_jpg = species_dest.with_suffix(".jpg")
if host_source.exists():
    shutil.move(host_source, host_dest)
if host_source_jpg.exists():
    shutil.move(host_source_jpg, host_dest_jpg)
if species_source.exists():
    shutil.move(species_source, species_dest)
if species_source_jpg.exists():
    shutil.move(species_source_jpg, species_dest_jpg)

wc.save_table(diagnostics, wc.table_path(context, 8, "patient_structure_diagnostics"))
wc.save_table(host_effects, wc.table_path(context, 9, "host_mixed_effects"))
wc.save_table(host_status, wc.table_path(context, 10, "host_mixed_status"))
wc.save_table(species_effects, wc.table_path(context, 11, "species_mixed_effects"))
wc.save_table(species_status, wc.table_path(context, 12, "species_mixed_status"))
wc.save_table(comparison, wc.table_path(context, 13, "mixed_vs_cluster_comparison"))

if advanced_context.input_dir.exists():
    shutil.rmtree(advanced_context.input_dir)


# %% [markdown]
# ## Review Numbered Outputs
#

# %%
diagnostics = pd.read_csv(
    wc.table_path(context, 8, "patient_structure_diagnostics"), sep="\t"
)
host_effects = pd.read_csv(wc.table_path(context, 9, "host_mixed_effects"), sep="\t")
host_status = pd.read_csv(wc.table_path(context, 10, "host_mixed_status"), sep="\t")
species_effects = pd.read_csv(
    wc.table_path(context, 11, "species_mixed_effects"), sep="\t"
)
comparison = pd.read_csv(
    wc.table_path(context, 13, "mixed_vs_cluster_comparison"), sep="\t"
)

display(diagnostics)
display(SVG(filename=str(wc.figure_path(context, 5, "host_model_compare"))))
display(SVG(filename=str(wc.figure_path(context, 6, "species_mixed_effects"))))
display(species_effects.head(20))

host_model = host_status.loc[
    (host_status["record_type"] == "model_fit")
    & (host_status["model_name"] == advanced.HOST_MODEL_NAME)
].iloc[0]
host_model_tests = host_status.loc[
    (host_status["record_type"] == "random_effect_test")
    & (host_status["model_name"] == advanced.HOST_MODEL_NAME)
].copy()
sign_agree = comparison["same_direction"].dropna().mean()
patient_test = host_model_tests.loc[
    host_model_tests["tested_effect"] == "patient_only_vs_fixed"
].iloc[0]
batch_test = host_model_tests.loc[
    host_model_tests["tested_effect"] == "batch_only_vs_fixed"
].iloc[0]
add_batch_test = host_model_tests.loc[
    host_model_tests["tested_effect"] == "batch_added_to_patient"
].iloc[0]
summary_lines = [
    f"- Patient-plus-batch host model AIC: {host_model['aic']:.2f}; patient variance {host_model['patient_var']:.3g}, batch variance {host_model['batch_var']:.3g}.",
    f"- Gaussian random-effect tests: patient vs fixed p={patient_test['pvalue_boundary']:.3g}, batch vs fixed p={batch_test['pvalue_boundary']:.3g}, add batch on top of patient p={add_batch_test['pvalue_boundary']:.3g}.",
    f"- Positive result: sign agreement between simple and mixed models was {sign_agree:.1%} across matched terms.",
    "- Positive result: P. aeruginosa chronic-like enrichment and S. aureus head/neck enrichment persisted in the mixed model.",
    "- Negative result: host fixed effects remain modest once patient and culture-date batch structure are modeled explicitly.",
    "- Negative result: single-group random-intercept fits were less informative than the joint patient-plus-batch specification.",
]
display(Markdown("## Working Interpretation\n" + "\n".join(summary_lines)))
