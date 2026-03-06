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
# # 07. Bracken Versus MetaPhlAn Sensitivity Analysis
#
# This notebook asks whether the main community and species-model conclusions are robust to the choice of bacterial profiling table.
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
# ## Define The Matched-Method Comparison Helpers
#

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, spearmanr

BODY_REGION_COMPARISON_ORDER = [
    "Same patient, same batch date",
    "Same patient, different batch date",
    "Different patient",
]

SHARED_SENSITIVITY_SPECIES = [
    "Staphylococcus aureus",
    "Pseudomonas aeruginosa",
    "Cutibacterium acnes",
    "Corynebacterium striatum",
    "Serratia marcescens",
]


def summarize_pairwise_groups(pairwise):
    summary = (
        pairwise.groupby("comparison_group")["distance"]
        .agg(["count", "median", "mean"])
        .reindex(BODY_REGION_COMPARISON_ORDER)
        .reset_index()
    )
    reference = pairwise.loc[pairwise["comparison_group"] == "Different patient", "distance"]
    rows = []
    for group in BODY_REGION_COMPARISON_ORDER[:-1]:
        test = pairwise.loc[pairwise["comparison_group"] == group, "distance"]
        if test.empty or reference.empty:
            continue
        statistic = mannwhitneyu(test, reference, alternative="less")
        rows.append(
            {
                "comparison_group": group,
                "reference_group": "Different patient",
                "pvalue": statistic.pvalue,
                "median_difference": test.median() - reference.median(),
            }
        )
    pvalues = base.bh_adjust(pd.DataFrame(rows), "pvalue") if rows else pd.DataFrame()
    return summary.merge(pvalues, on="comparison_group", how="left")


def fit_method_species_models(qc, counts, species_list):
    sample_ids = qc.index.tolist()
    counts = counts.loc[sample_ids].copy()
    counts = counts.loc[:, counts.sum(axis=0) > 0]
    clr = base.clr_transform(counts)
    model_df = qc.loc[
        sample_ids,
        ["patient_id", "body_region", "chronicity_group", "log10_bacterial_reads"],
    ].copy()
    rows = []
    for species in species_list:
        if species not in clr.columns:
            continue
        prevalence = float((counts[species] > 0).mean())
        if prevalence < 0.1:
            continue
        frame = model_df.copy()
        frame["response"] = clr[species]
        formula = (
            "response ~ C(body_region, Treatment('lower_extremity')) "
            "+ C(chronicity_group, Treatment('unknown')) + log10_bacterial_reads"
        )
        fit = smf.ols(formula, data=frame).fit(
            cov_type="cluster",
            cov_kwds={"groups": frame["patient_id"]},
        )
        conf = fit.conf_int()
        for term, estimate in fit.params.items():
            if term == "Intercept":
                continue
            rows.append(
                {
                    "species": species,
                    "term": term,
                    "estimate": estimate,
                    "conf_low": conf.loc[term, 0],
                    "conf_high": conf.loc[term, 1],
                    "pvalue": fit.pvalues[term],
                    "prevalence": prevalence,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["species", "term", "estimate", "conf_low", "conf_high", "pvalue", "prevalence", "qvalue"]
        )
    return base.bh_adjust(pd.DataFrame(rows).sort_values("pvalue"), "pvalue")


def build_bracken_metaphlan_tables():
    qc = base_data["qc"]
    species_bac = base_data["species_bac"]
    metaphlan_species = base_data["metaphlan_species"]

    sample_depth = pd.DataFrame(
        {
            "sample_id": qc.index,
            "patient_id": qc["patient_id"].values,
            "visit_id": qc["visit_id"].values,
            "host_removed_fraction": qc["host_removed_fraction"].values,
            "bracken_bacterial_species_reads": qc["bacterial_species_reads"].values,
            "metaphlan_species_reads": metaphlan_species.sum(axis=1).reindex(qc.index).fillna(0).values,
            "bracken_qc_pass": qc["community_qc_pass"].values,
        }
    )
    sample_depth["metaphlan_nonzero"] = sample_depth["metaphlan_species_reads"] > 0
    sample_depth["shared_method_qc"] = sample_depth["bracken_qc_pass"] & sample_depth["metaphlan_nonzero"]

    shared_samples = sample_depth.loc[sample_depth["shared_method_qc"], "sample_id"].tolist()
    metadata_subset = base_data["metadata"].loc[shared_samples]

    bracken_rel = species_bac.loc[shared_samples].copy()
    bracken_rel = bracken_rel.loc[:, bracken_rel.sum(axis=0) > 0]
    bracken_rel = bracken_rel.div(bracken_rel.sum(axis=1), axis=0)

    metaphlan_rel = metaphlan_species.loc[shared_samples].copy()
    metaphlan_rel = metaphlan_rel.loc[:, metaphlan_rel.sum(axis=0) > 0]
    metaphlan_rel = metaphlan_rel.div(metaphlan_rel.sum(axis=1), axis=0)

    bracken_pairwise = base.summarize_pairwise_distances(bracken_rel, metadata_subset)
    metaphlan_pairwise = base.summarize_pairwise_distances(metaphlan_rel, metadata_subset)

    bracken_summary = summarize_pairwise_groups(bracken_pairwise).assign(method="Bracken")
    metaphlan_summary = summarize_pairwise_groups(metaphlan_pairwise).assign(method="MetaPhlAn")
    distance_summary = pd.concat([bracken_summary, metaphlan_summary], ignore_index=True)

    shared_species = [
        species
        for species in SHARED_SENSITIVITY_SPECIES
        if species in bracken_rel.columns and species in metaphlan_rel.columns
    ]
    taxon_rows = []
    for species in shared_species:
        bracken_values = np.log10(bracken_rel[species] + 1e-6)
        metaphlan_values = np.log10(metaphlan_rel[species] + 1e-6)
        rho, pvalue = spearmanr(bracken_values, metaphlan_values)
        taxon_rows.append(
            {
                "species": species,
                "spearman_rho": rho,
                "pvalue": pvalue,
                "bracken_prevalence": float((bracken_rel[species] > 0).mean()),
                "metaphlan_prevalence": float((metaphlan_rel[species] > 0).mean()),
            }
        )
    taxon_correlations = base.bh_adjust(pd.DataFrame(taxon_rows).sort_values("pvalue"), "pvalue")

    qc_subset = qc.loc[shared_samples].copy()
    bracken_models = fit_method_species_models(qc_subset, species_bac.loc[shared_samples], shared_species)
    metaphlan_models = fit_method_species_models(qc_subset, metaphlan_species.loc[shared_samples], shared_species)

    model_comparison = bracken_models.merge(
        metaphlan_models,
        on=["species", "term"],
        suffixes=("_bracken", "_metaphlan"),
        how="inner",
    )
    model_comparison["same_direction"] = np.sign(model_comparison["estimate_bracken"]) == np.sign(
        model_comparison["estimate_metaphlan"]
    )
    model_comparison["term_label"] = model_comparison["term"].map(base.prettify_model_term)
    return {
        "sample_depth": sample_depth,
        "distance_summary": distance_summary,
        "taxon_correlations": taxon_correlations,
        "bracken_models": bracken_models,
        "metaphlan_models": metaphlan_models,
        "model_comparison": model_comparison.sort_values(["species", "term"]),
    }


def make_bracken_metaphlan_figure(comparison_data):
    sample_depth = comparison_data["sample_depth"]
    distance_summary = comparison_data["distance_summary"]
    taxon_correlations = comparison_data["taxon_correlations"]
    model_comparison = comparison_data["model_comparison"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    plot_df = sample_depth.loc[sample_depth["shared_method_qc"]].copy()
    sns.scatterplot(
        data=plot_df,
        x="bracken_bacterial_species_reads",
        y="metaphlan_species_reads",
        hue="host_removed_fraction",
        palette="viridis",
        s=65,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Bracken bacterial species reads")
    ax.set_ylabel("MetaPhlAn species reads")
    ax.set_title("Sample-level bacterial depth agreement")

    ax = axes[0, 1]
    plot_summary = distance_summary.copy()
    plot_summary["comparison_group"] = pd.Categorical(
        plot_summary["comparison_group"],
        categories=BODY_REGION_COMPARISON_ORDER,
        ordered=True,
    )
    sns.pointplot(
        data=plot_summary.sort_values("comparison_group"),
        x="comparison_group",
        y="median",
        hue="method",
        dodge=0.2,
        markers=["o", "s"],
        linestyles=["-", "--"],
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Median Bray-Curtis distance")
    ax.set_title("Within-patient structure by method")
    ax.tick_params(axis="x", rotation=15)

    ax = axes[1, 0]
    if not taxon_correlations.empty:
        plot_corr = taxon_correlations.sort_values("spearman_rho")
        ax.barh(plot_corr["species"], plot_corr["spearman_rho"], color="#3b6a8f")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Spearman rho")
    else:
        ax.text(0.5, 0.5, "No shared taxa available", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("Shared-taxon abundance concordance")

    ax = axes[1, 1]
    if not model_comparison.empty:
        sns.scatterplot(
            data=model_comparison,
            x="estimate_bracken",
            y="estimate_metaphlan",
            hue="species",
            s=75,
            ax=ax,
        )
        low = np.nanmin(
            [model_comparison["estimate_bracken"].min(), model_comparison["estimate_metaphlan"].min()]
        ) - 0.2
        high = np.nanmax(
            [model_comparison["estimate_bracken"].max(), model_comparison["estimate_metaphlan"].max()]
        ) + 0.2
        ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel("Bracken coefficient")
        ax.set_ylabel("MetaPhlAn coefficient")
    else:
        ax.text(0.5, 0.5, "No matched model terms", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("Shared model-term comparison")

    fig.tight_layout()
    out_path = wc.figure_path(context, 7, "bracken_metaphlan_sensitivity")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



# %% [markdown]
# ## Run The Sensitivity Analysis And Save Numbered Outputs
#

# %%
comparison_data = build_bracken_metaphlan_tables()
make_bracken_metaphlan_figure(comparison_data)

wc.save_table(comparison_data["sample_depth"], wc.table_path(context, 14, "bracken_metaphlan_sample_depth"))
wc.save_table(comparison_data["distance_summary"], wc.table_path(context, 15, "bracken_metaphlan_distance_summary"))
wc.save_table(comparison_data["taxon_correlations"], wc.table_path(context, 16, "bracken_metaphlan_taxon_correlations"))
wc.save_table(comparison_data["bracken_models"], wc.table_path(context, 17, "bracken_species_models_shared_samples"))
wc.save_table(comparison_data["metaphlan_models"], wc.table_path(context, 18, "metaphlan_species_models"))
wc.save_table(comparison_data["model_comparison"], wc.table_path(context, 19, "bracken_metaphlan_model_comparison"))


# %% [markdown]
# ## Review Numbered Outputs
#

# %%
sample_depth = pd.read_csv(wc.table_path(context, 14, "bracken_metaphlan_sample_depth"), sep="\t")
distance_summary = pd.read_csv(wc.table_path(context, 15, "bracken_metaphlan_distance_summary"), sep="\t")
taxon_correlations = pd.read_csv(wc.table_path(context, 16, "bracken_metaphlan_taxon_correlations"), sep="\t")
model_comparison = pd.read_csv(wc.table_path(context, 19, "bracken_metaphlan_model_comparison"), sep="\t")

display(SVG(filename=str(wc.figure_path(context, 7, "bracken_metaphlan_sensitivity"))))
display(distance_summary)
display(taxon_correlations)
display(model_comparison.head(20))

shared_n = int(sample_depth["shared_method_qc"].sum())
same_direction = model_comparison["same_direction"].mean() if not model_comparison.empty else float("nan")
summary_lines = [
    f"- Samples retained for direct method comparison: {shared_n}.",
    "- Positive result: the major sample-structure pattern can be compared on a matched sample set rather than by memory or intuition.",
    f"- Positive / negative result: matched Bracken-vs-MetaPhlAn coefficient directions agreed in {same_direction:.1%} of shared terms." if pd.notna(same_direction) else "- No matched model terms were available for direction comparison.",
    "- Negative result: MetaPhlAn is sparser here, so some Bracken-supported signals are expected to attenuate rather than replicate perfectly.",
]
display(Markdown("## Working Interpretation\n" + "\n".join(summary_lines)))

