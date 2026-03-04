#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

try:
    import analysis_update_base_kernel as base
except ImportError:  # pragma: no cover - direct script execution fallback
    import analysis_base as base


HOST_FORMULAS = {
    "host_base": (
        "host_logit ~ C(body_region, Treatment('lower_extremity')) "
        "+ C(culture_positive_label, Treatment('negative')) + years_since_first_sample"
    ),
    "host_extended": (
        "host_logit ~ C(body_region, Treatment('lower_extremity')) "
        "+ C(chronicity_group, Treatment('unknown')) "
        "+ C(culture_positive_label, Treatment('negative')) + years_since_first_sample"
    ),
}

SPECIES_FORMULA = (
    "response ~ C(body_region, Treatment('lower_extremity')) "
    "+ C(chronicity_group, Treatment('unknown')) + log10_bacterial_reads + years_since_first_sample"
)

ADVANCED_SPECIES = [
    "Staphylococcus aureus",
    "Pseudomonas aeruginosa",
    "Serratia marcescens",
    "Klebsiella pneumoniae",
    "Cutibacterium acnes",
    "Corynebacterium striatum",
]


@dataclass
class AdvancedContext:
    data_dir: Path
    output_dir: Path
    figure_dir: Path
    table_dir: Path
    input_dir: Path


def ensure_dirs(context: AdvancedContext) -> None:
    context.output_dir.mkdir(exist_ok=True)
    context.figure_dir.mkdir(exist_ok=True)
    context.table_dir.mkdir(exist_ok=True)
    context.input_dir.mkdir(exist_ok=True)


def prepare_base_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_context = base.AnalysisContext(
        data_dir=data_dir,
        analysis_dir=data_dir / "analysis_update",
        figure_dir=data_dir / "analysis_update" / "figures",
        table_dir=data_dir / "analysis_update" / "tables",
    )
    species_all, species_bac = base.load_bracken_tables(base_context)
    metaphlan = base.load_metaphlan_table(base_context)
    sample_ids = sorted(species_all.index)
    metadata = base.load_metadata(base_context, sample_ids)
    qc = base.prepare_qc_table(base_context, metadata, species_all, species_bac, metaphlan)
    qc["patient_id"] = qc["patient_id"].astype(str)
    qc["visit_id"] = qc["visit_id"].astype(str)
    qc["batch_id"] = qc["batch_id"].astype(str)
    return qc, species_all, species_bac


def fit_variance_component_model(
    data: pd.DataFrame,
    formula: str,
    outcome_name: str,
    model_name: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = data.copy()
    frame["all_group"] = "all"
    vc_formula = {"patient": "0 + C(patient_id)", "batch": "0 + C(batch_id)"}
    candidates: list[tuple[object, list[str], str]] = []
    last_error = None

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
            warning_messages = [str(item.message) for item in caught]
            candidates.append((fit, warning_messages, method))
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc

    if not candidates:
        raise RuntimeError(
            f"Mixed model failed for {outcome_name} [{model_name}]: {last_error!r}"
        ) from last_error

    converged_candidates = [item for item in candidates if bool(getattr(item[0], "converged", False))]
    pool = converged_candidates if converged_candidates else candidates
    fit, warning_messages, optimizer = min(pool, key=lambda item: item[0].aic)

    fixed_terms = fit.fe_params.index.tolist()
    conf = fit.conf_int()
    rows = []
    for term in fixed_terms:
        rows.append(
            {
                "outcome": outcome_name,
                "model_name": model_name,
                "term": term,
                "estimate": fit.fe_params[term],
                "conf_low": conf.loc[term, 0],
                "conf_high": conf.loc[term, 1],
                "pvalue": fit.pvalues.get(term, np.nan),
            }
        )

    result_table = pd.DataFrame(rows)
    status = {
        "outcome": outcome_name,
        "model_name": model_name,
        "n_samples": frame.shape[0],
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
    return result_table, status


def fit_host_models(qc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_df = qc.dropna(
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
    effect_tables = []
    status_rows = []
    for model_name, formula in HOST_FORMULAS.items():
        effects, status = fit_variance_component_model(model_df, formula, "host_logit", model_name)
        effect_tables.append(effects)
        status_rows.append(status)
    effects_df = pd.concat(effect_tables, ignore_index=True)
    effects_df = base.bh_adjust(effects_df.sort_values(["model_name", "pvalue"]), "pvalue")
    status_df = pd.DataFrame(status_rows)
    return effects_df, status_df


def fit_species_models(qc: pd.DataFrame, species_bac: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_ids = qc.index[qc["model_qc_pass"]].tolist()
    counts = species_bac.loc[sample_ids].copy()
    clr = base.clr_transform(counts)

    metadata = qc.loc[
        sample_ids,
        [
            "patient_id",
            "batch_id",
            "body_region",
            "chronicity_group",
            "log10_bacterial_reads",
            "years_since_first_sample",
        ],
    ].copy()

    effect_tables = []
    status_rows = []

    for species in ADVANCED_SPECIES:
        if species not in clr.columns:
            continue
        prevalence = float((counts[species] > 0).mean())
        if prevalence < 0.1:
            continue
        frame = metadata.copy()
        frame["response"] = clr[species]
        effects, status = fit_variance_component_model(frame, SPECIES_FORMULA, species, "species_mixedlm")
        effects["prevalence"] = prevalence
        status["prevalence"] = prevalence
        effect_tables.append(effects)
        status_rows.append(status)

    effects_df = pd.concat(effect_tables, ignore_index=True)
    effects_df = base.bh_adjust(effects_df.sort_values("pvalue"), "pvalue")
    status_df = pd.DataFrame(status_rows).sort_values("aic")
    return effects_df, status_df


def prepare_external_tool_inputs(
    context: AdvancedContext,
    qc: pd.DataFrame,
    species_bac: pd.DataFrame,
) -> None:
    sample_ids = qc.index[qc["model_qc_pass"]].tolist()
    rel_ab = species_bac.loc[sample_ids].copy()
    rel_ab = rel_ab.div(rel_ab.sum(axis=1), axis=0).fillna(0)
    prevalence = (rel_ab > 0).mean(axis=0)
    rel_ab = rel_ab.loc[:, prevalence >= 0.1]

    metadata = qc.loc[
        sample_ids,
        [
            "patient_id",
            "visit_id",
            "culture_date",
            "batch_id",
            "body_region",
            "chronicity_group",
            "clinical_infection_flag",
            "host_removed_fraction",
            "log10_bacterial_reads",
            "years_since_first_sample",
        ]
        + [f"culture_{cfg['group']}" for cfg in base.CULTURE_GROUPS],
    ].copy()
    metadata.index.name = "sample_id"

    rel_ab.to_csv(context.input_dir / "maaslin2_samples_by_features.tsv", sep="\t")
    rel_ab.transpose().to_csv(context.input_dir / "maaslin2_features_by_samples.tsv", sep="\t")
    metadata.to_csv(context.input_dir / "maaslin2_metadata.tsv", sep="\t")

    halla_meta = pd.get_dummies(
        metadata.drop(columns=["culture_date"]),
        columns=["body_region", "chronicity_group", "clinical_infection_flag"],
        dtype=float,
    )
    halla_meta.to_csv(context.input_dir / "halla_metadata_numeric.tsv", sep="\t")
    rel_ab.to_csv(context.input_dir / "halla_microbiome_samples_by_features.tsv", sep="\t")

    top_taxa = rel_ab.mean(axis=0).sort_values(ascending=False).head(50).index.tolist()
    lefse = pd.DataFrame(index=["class", "subclass", "subject"] + top_taxa, columns=sample_ids)
    lefse.loc["class"] = metadata["body_region"]
    lefse.loc["subclass"] = metadata["chronicity_group"]
    lefse.loc["subject"] = metadata["patient_id"]
    for taxon in top_taxa:
        lefse.loc[taxon] = rel_ab[taxon]
    lefse.to_csv(context.input_dir / "lefse_body_region_input.tsv", sep="\t")


def assess_optional_methods(context: AdvancedContext) -> pd.DataFrame:
    rows = [
        {
            "method": "statsmodels_mixedlm",
            "status": "ran",
            "runnable": True,
            "reason": "Python variance-component mixed models executed successfully.",
            "detail": "",
        }
    ]

    try:
        result = subprocess.run(
            ["halla", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            rows.append(
                {
                    "method": "HAllA",
                    "status": "available_but_not_run",
                    "runnable": True,
                    "reason": "CLI appears available. Use prepared inputs for exploratory association testing.",
                    "detail": "",
                }
            )
        else:
            stderr = (result.stderr or result.stdout).strip()
            if "XICOR" in stderr:
                detail = "Missing R package XICOR required by the installed HAllA entrypoint."
            else:
                detail = stderr.replace("\n", " ")[:1000]
            rows.append(
                {
                    "method": "HAllA",
                    "status": "blocked_dependency",
                    "runnable": False,
                    "reason": "Installed CLI fails to start in the current environment.",
                    "detail": detail,
                }
            )
    except FileNotFoundError:
        rows.append(
            {
                "method": "HAllA",
                "status": "missing",
                "runnable": False,
                "reason": "CLI not installed.",
                "detail": "",
            }
        )

    for method, reason in [
        (
            "MaAsLin2",
            "R package is not installed locally. Inputs were prepared because this is a strong follow-up multivariable method for this dataset.",
        ),
        (
            "LEfSe",
            "Executable/package is not installed locally. Inputs were prepared, but the method is less suitable here because it does not model repeated measures and multiple covariates as directly as mixed models or MaAsLin2.",
        ),
    ]:
        rows.append(
            {
                "method": method,
                "status": "missing",
                "runnable": False,
                "reason": reason,
                "detail": "",
            }
        )

    return pd.DataFrame(rows)


def load_cluster_results(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    host_cluster = pd.read_csv(
        data_dir / "analysis_update" / "tables" / "table_02_02_host_model.tsv",
        sep="\t",
    )
    host_cluster["outcome"] = "host_logit"
    host_cluster["model_name"] = "cluster_ols"

    species_cluster = pd.read_csv(
        data_dir / "analysis_update" / "tables" / "table_05_01_species_associations.tsv",
        sep="\t",
    )
    species_cluster["model_name"] = "cluster_ols"
    return host_cluster, species_cluster


def build_comparison_table(
    host_mixed: pd.DataFrame,
    species_mixed: pd.DataFrame,
    host_cluster: pd.DataFrame,
    species_cluster: pd.DataFrame,
) -> pd.DataFrame:
    mixed = pd.concat(
        [
            host_mixed.loc[:, ["outcome", "term", "estimate", "pvalue", "qvalue"]].assign(model_family="mixedlm"),
            species_mixed.loc[:, ["outcome", "term", "estimate", "pvalue", "qvalue"]].assign(model_family="mixedlm"),
        ],
        ignore_index=True,
    )
    cluster = pd.concat(
        [
            host_cluster.loc[:, ["outcome", "term", "estimate", "pvalue", "qvalue"]].assign(model_family="cluster_ols"),
            species_cluster.loc[:, ["species", "term", "estimate", "pvalue", "qvalue"]]
            .rename(columns={"species": "outcome"})
            .assign(model_family="cluster_ols"),
        ],
        ignore_index=True,
    )
    comparison = mixed.merge(
        cluster,
        on=["outcome", "term"],
        suffixes=("_mixed", "_cluster"),
        how="outer",
    )
    comparison["same_direction"] = np.sign(comparison["estimate_mixed"]) == np.sign(comparison["estimate_cluster"])
    comparison["estimate_shift"] = comparison["estimate_mixed"] - comparison["estimate_cluster"]
    return comparison.sort_values(["outcome", "term"])


def make_host_comparison_figure(
    host_mixed: pd.DataFrame,
    host_cluster: pd.DataFrame,
    context: AdvancedContext,
) -> None:
    mixed = host_mixed.loc[host_mixed["model_name"] == "host_base"].copy()
    cluster = host_cluster.copy()
    data = mixed.merge(
        cluster[["term", "estimate"]].rename(columns={"estimate": "estimate_cluster"}),
        on="term",
        how="inner",
    )
    data = data.loc[data["term"] != "Intercept"].copy()
    data["term_label"] = data["term"].map(base.prettify_model_term)
    data = data.sort_values("estimate")

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.6 * data.shape[0])))
    y = np.arange(data.shape[0])
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    for _, row in data.iterrows():
        idx = data.index.get_loc(row.name)
        ax.plot(
            [row["estimate_cluster"], row["estimate"]],
            [idx, idx],
            color="#9eb6d8",
            linewidth=2,
        )
    ax.scatter(data["estimate_cluster"], y, color="#7f7f7f", label="Cluster-robust OLS", s=60)
    ax.scatter(data["estimate"], y, color="#1f4e79", label="Variance-component mixed model", s=60)
    ax.set_yticks(y)
    ax.set_yticklabels(data["term_label"])
    ax.set_xlabel("Coefficient estimate")
    ax.set_ylabel("")
    ax.set_title("Host model: cluster-robust versus mixed-effects estimates")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(context.figure_dir / "advanced_figure_01_host_compare.svg", bbox_inches="tight")
    plt.close(fig)


def make_species_mixed_figure(species_mixed: pd.DataFrame, context: AdvancedContext) -> pd.DataFrame:
    plot_df = species_mixed.loc[
        species_mixed["term"].str.contains("body_region") | species_mixed["term"].str.contains("chronicity_group")
    ].copy()
    plot_df = plot_df.loc[plot_df["qvalue"].fillna(1) <= 0.15].copy()
    if plot_df.empty:
        plot_df = species_mixed.nsmallest(12, "pvalue").copy()

    plot_df["term_label"] = plot_df["term"].map(base.prettify_model_term)
    plot_df["y_label"] = plot_df["outcome"] + " | " + plot_df["term_label"]
    plot_df = plot_df.sort_values("estimate")

    fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * plot_df.shape[0])))
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.errorbar(
        x=plot_df["estimate"],
        y=np.arange(plot_df.shape[0]),
        xerr=[
            plot_df["estimate"] - plot_df["conf_low"],
            plot_df["conf_high"] - plot_df["estimate"],
        ],
        fmt="o",
        color="#7c3f00",
        ecolor="#d8a36c",
        capsize=3,
    )
    ax.set_yticks(np.arange(plot_df.shape[0]))
    ax.set_yticklabels(plot_df["y_label"])
    ax.set_xlabel("Mixed-model fixed effect")
    ax.set_ylabel("")
    ax.set_title("Selected variance-component mixed-model associations")
    fig.tight_layout()
    fig.savefig(context.figure_dir / "advanced_figure_02_species_mixed.svg", bbox_inches="tight")
    plt.close(fig)
    return plot_df


def write_report(
    context: AdvancedContext,
    host_effects: pd.DataFrame,
    host_status: pd.DataFrame,
    species_effects: pd.DataFrame,
    species_status: pd.DataFrame,
    method_status: pd.DataFrame,
    comparison: pd.DataFrame,
    species_plot_df: pd.DataFrame,
) -> None:
    host_base = host_effects.loc[host_effects["model_name"] == "host_base"].copy()
    host_best = host_base.loc[host_base["term"] != "Intercept"].sort_values("pvalue").head(1)
    host_ext = host_effects.loc[host_effects["model_name"] == "host_extended"].copy()
    host_ext_best = host_ext.loc[host_ext["term"].str.contains("chronicity_group")].sort_values("pvalue").head(1)
    species_hits = species_effects.sort_values("qvalue").head(6)
    mixed_status = species_status.loc[:, ["outcome", "aic", "patient_var", "batch_var", "warning_count"]]
    agree = comparison["same_direction"].dropna().mean()

    lines = [
        "# Advanced mixed-effects layer",
        "",
        "## Summary",
        "",
        "- These models add variance components for patient and culture-date batch on top of the original fixed-effect analyses.",
        "- Patient-only mixed models were often singular in this cohort, so the working specification uses variance components for both patient and batch with a constant grouping factor.",
        f"- Sign agreement between cluster-robust and mixed-model estimates was {agree:.1%} across terms that were estimable in both frameworks.",
        "",
        "## Host models",
        "",
    ]

    if (host_base["qvalue"] <= 0.1).any() and not host_best.empty:
        row = host_best.iloc[0]
        lines.append(
            f"- Base host mixed model: strongest non-intercept term was `{base.prettify_model_term(row['term'])}` with estimate {row['estimate']:.2f} (q={row['qvalue']:.3g})."
        )
    elif not host_best.empty:
        row = host_best.iloc[0]
        lines.append(
            f"- Base host mixed model: no fixed-effect term reached q <= 0.1; the largest signal was `{base.prettify_model_term(row['term'])}` with estimate {row['estimate']:.2f} (q={row['qvalue']:.3g})."
        )
    if not host_ext_best.empty:
        row = host_ext_best.iloc[0]
        lines.append(
            f"- Extended host mixed model: chronicity signal `{base.prettify_model_term(row['term'])}` had estimate {row['estimate']:.2f} (q={row['qvalue']:.3g})."
        )

    host_status_row = host_status.loc[host_status["model_name"] == "host_base"].iloc[0]
    lines.append(
        f"- Host base model variance components: patient {host_status_row['patient_var']:.3g}, batch {host_status_row['batch_var']:.3g}."
    )

    lines.extend(["", "## Species models", ""])
    for _, row in species_hits.iterrows():
        lines.append(
            f"- {row['outcome']}: {base.prettify_model_term(row['term'])} -> {row['estimate']:.2f} (95% CI {row['conf_low']:.2f} to {row['conf_high']:.2f}, q={row['qvalue']:.3g})."
        )

    lines.extend(
        [
            "",
            "## Method assessment",
            "",
        ]
    )
    for _, row in method_status.iterrows():
        detail = f" Detail: {row['detail']}" if row["detail"] else ""
        lines.append(f"- {row['method']}: {row['status']}. {row['reason']}{detail}")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- MaAsLin2 remains the best external follow-up for multivariable microbiome association once the R package is installed in the `eb` environment.",
            "- LEfSe was considered but kept as a prepared-input option only, because it is fundamentally less aligned with repeated-measures multi-covariate inference.",
            "- HAllA is potentially useful for exploratory microbe-metadata block associations, but the local installation is blocked by a missing R dependency (`XICOR`).",
            "",
            "## Figures",
            "",
            "1. `advanced_figure_01_host_compare.svg`: Host-fraction coefficient comparison between the original cluster-robust regression and the advanced variance-component mixed model.",
            "2. `advanced_figure_02_species_mixed.svg`: Mixed-model fixed effects for selected taxa, emphasizing body-region and chronicity terms.",
            "",
            "## Tables",
            "",
            "- `host_mixed_effects.tsv` and `host_mixed_status.tsv` summarize host mixed models.",
            "- `species_mixed_effects.tsv` and `species_mixed_status.tsv` summarize taxon mixed models.",
            "- `mixed_vs_cluster_comparison.tsv` compares advanced-model coefficients against the earlier cluster-robust estimates.",
            "- `method_status.tsv` records which optional tools ran and which were blocked by missing dependencies.",
            "",
            "## Species shown in Figure 2",
            "",
        ]
    )
    for _, row in species_plot_df.iterrows():
        lines.append(
            f"- {row['outcome']}: {row['term_label']} -> {row['estimate']:.2f} (95% CI {row['conf_low']:.2f} to {row['conf_high']:.2f})."
        )

    (context.output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run advanced mixed-effects models for EB metagenomics.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the metagenomics_20260206 directory.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    context = AdvancedContext(
        data_dir=data_dir,
        output_dir=data_dir / "analysis_update" / "advanced",
        figure_dir=data_dir / "analysis_update" / "advanced" / "figures",
        table_dir=data_dir / "analysis_update" / "advanced" / "tables",
        input_dir=data_dir / "analysis_update" / "advanced" / "inputs",
    )
    ensure_dirs(context)

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42

    qc, _, species_bac = prepare_base_data(data_dir)
    host_effects, host_status = fit_host_models(qc)
    species_effects, species_status = fit_species_models(qc, species_bac)
    prepare_external_tool_inputs(context, qc, species_bac)
    method_status = assess_optional_methods(context)

    host_cluster, species_cluster = load_cluster_results(data_dir)
    comparison = build_comparison_table(
        host_effects,
        species_effects,
        host_cluster,
        species_cluster,
    )

    make_host_comparison_figure(host_effects, host_cluster, context)
    species_plot_df = make_species_mixed_figure(species_effects, context)

    host_effects.to_csv(context.table_dir / "host_mixed_effects.tsv", sep="\t", index=False)
    host_status.to_csv(context.table_dir / "host_mixed_status.tsv", sep="\t", index=False)
    species_effects.to_csv(context.table_dir / "species_mixed_effects.tsv", sep="\t", index=False)
    species_status.to_csv(context.table_dir / "species_mixed_status.tsv", sep="\t", index=False)
    comparison.to_csv(context.table_dir / "mixed_vs_cluster_comparison.tsv", sep="\t", index=False)
    method_status.to_csv(context.table_dir / "method_status.tsv", sep="\t", index=False)

    write_report(
        context=context,
        host_effects=host_effects,
        host_status=host_status,
        species_effects=species_effects,
        species_status=species_status,
        method_status=method_status,
        comparison=comparison,
        species_plot_df=species_plot_df,
    )


if __name__ == "__main__":
    main()
