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
# # 04. Culture Concordance
#
# This notebook compares culture calls against Bracken relative abundance for clinically important organism groups.
# Descriptive inference is centered on Mann-Whitney U tests, with AUROC reported as a companion discrimination summary.
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
# ## Build Culture-Concordance Summaries
#

# %%
analysis_context = wc.base_analysis_context(context)
qc = base_data["qc"]
species_bac = base_data["species_bac"]

culture_summary, culture_plot_df = base.make_culture_abundance_table(qc, species_bac)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

top_labels = culture_summary.head(6)["label"].tolist()
culture_plot_df = culture_plot_df.loc[culture_plot_df["label"].isin(top_labels)].copy()
culture_plot_df["culture_status"] = np.where(culture_plot_df["culture_positive"], "Culture positive", "Culture negative")
culture_plot_df["log10_relative_abundance"] = np.log10(culture_plot_df["relative_abundance"] + 1e-6)

fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
axes = axes.ravel()
for ax, label in zip(axes, top_labels):
    data = culture_plot_df.loc[culture_plot_df["label"] == label]
    sns.boxplot(
        data=data,
        x="culture_status",
        y="log10_relative_abundance",
        color="#f2d4b7",
        width=0.6,
        fliersize=0,
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x="culture_status",
        y="log10_relative_abundance",
        color="#5b3417",
        alpha=0.5,
        size=4,
        ax=ax,
    )
    metrics = culture_summary.loc[culture_summary["label"] == label].iloc[0]
    ax.set_title(f"{label}\nU-test q={metrics['qvalue']:.3g}; AUROC={metrics['auroc']:.2f}")
    ax.set_xlabel("")
    ax.set_ylabel("log10(relative abundance + 1e-6)")
    ax.tick_params(axis="x", rotation=20)
for ax in axes[len(top_labels):]:
    ax.axis("off")
fig.tight_layout()
fig_path = wc.figure_path(context, 3, "culture_concordance")
fig.savefig(fig_path, bbox_inches="tight")
fig.savefig(fig_path.with_suffix(".jpg"), bbox_inches="tight", dpi=300)
plt.close(fig)

wc.save_table(culture_summary, wc.table_path(context, 6, "culture_concordance"))


# %% [markdown]
# ## Review Numbered Outputs
#

# %%
culture_summary = pd.read_csv(wc.table_path(context, 6, "culture_concordance"), sep="\t")

display(SVG(filename=str(wc.figure_path(context, 3, "culture_concordance"))))
display(culture_summary)

top_hit = culture_summary.sort_values("qvalue").iloc[0]
weakest = culture_summary.sort_values("qvalue", ascending=False).iloc[0]
summary_lines = [
    f"- Positive result: strongest descriptive agreement was {top_hit['label']} (Mann-Whitney U={top_hit['u_statistic']:.1f}, q={top_hit['qvalue']:.3g}, AUROC={top_hit['auroc']:.2f}).",
    f"- Negative result: weakest supported target here was {weakest['label']} (Mann-Whitney q={weakest['qvalue']:.3g}, AUROC={weakest['auroc']:.2f}).",
    "- Culture agreement is organism-group level, not strain-level or resistance-level agreement.",
]
display(Markdown("## Working Interpretation\n" + "\n".join(summary_lines)))

