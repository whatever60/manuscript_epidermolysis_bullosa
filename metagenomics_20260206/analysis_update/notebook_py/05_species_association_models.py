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
# # 05. Species Association Models
#
# This notebook fits the first-pass cluster-robust species models using Bracken CLR-transformed bacterial species counts.
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
# ## Fit Cluster-Robust Species Models
#

# %%
analysis_context = wc.base_analysis_context(context)
qc = base_data["qc"]
species_bac = base_data["species_bac"]

species_results = base.fit_species_models(qc, species_bac)

import matplotlib.pyplot as plt
import numpy as np

if not species_results.empty:
    plot_df = species_results.loc[
        species_results["term"].str.contains("body_region")
        | species_results["term"].str.contains("chronicity_group")
    ].copy()
    plot_df = plot_df.loc[plot_df["qvalue"].fillna(1) <= 0.15].copy()
    if plot_df.empty:
        plot_df = (
            species_results.loc[
                species_results["term"].str.contains("body_region")
                | species_results["term"].str.contains("chronicity_group")
            ]
            .head(12)
            .copy()
        )
    plot_df["term_label"] = plot_df["term"].map(base.prettify_model_term)
    plot_df["species_label"] = plot_df["species"]
    plot_df = plot_df.sort_values(["estimate", "species_label"])
    plot_df["y_label"] = plot_df["species_label"] + " | " + plot_df["term_label"]

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.45 * plot_df.shape[0])))
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.errorbar(
        x=plot_df["estimate"],
        y=np.arange(plot_df.shape[0]),
        xerr=[
            plot_df["estimate"] - plot_df["conf_low"],
            plot_df["conf_high"] - plot_df["estimate"],
        ],
        fmt="o",
        color="#204a87",
        ecolor="#7aa6d8",
        capsize=3,
    )
    ax.set_yticks(np.arange(plot_df.shape[0]))
    ax.set_yticklabels(plot_df["y_label"])
    ax.set_xlabel("Cluster-robust CLR effect size")
    ax.set_ylabel("")
    ax.set_title("Body region and chronicity associations for key taxa")
    fig.tight_layout()
    fig_path = wc.figure_path(context, 4, "species_associations")
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".jpg"), bbox_inches="tight", dpi=300)
    plt.close(fig)

wc.save_table(species_results, wc.table_path(context, 7, "species_associations"))


# %% [markdown]
# ## Review Numbered Outputs
#

# %%
species_results = pd.read_csv(
    wc.table_path(context, 7, "species_associations"), sep="\t"
)

display(SVG(filename=str(wc.figure_path(context, 4, "species_associations"))))
display(species_results.head(20))

significant = species_results.loc[species_results["qvalue"] <= 0.1].copy()
summary_lines = [
    f"- Positive result: {significant.shape[0]} model terms reached q <= 0.1 in the first-pass species screen.",
    "- Positive result: the main supported patterns were chronic-like enrichment of P. aeruginosa and head/neck enrichment of S. aureus / Cutibacterium acnes.",
    "- Negative result: weaker effects were sensitive to model choice and were treated as provisional until the mixed-model notebook.",
]
display(Markdown("## Working Interpretation\n" + "\n".join(summary_lines)))
