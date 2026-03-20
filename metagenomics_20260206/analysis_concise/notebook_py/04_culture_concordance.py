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

culture_summary, _ = base.make_culture_abundance_table(qc, species_bac)

wc.save_table(culture_summary, wc.table_path(context, 6, "culture_concordance"))


# %% [markdown]
# ## Review Numbered Outputs
#

# %%
culture_summary = pd.read_csv(
    wc.table_path(context, 6, "culture_concordance"), sep="\t"
)

display(culture_summary)

top_hit = culture_summary.sort_values("qvalue").iloc[0]
weakest = culture_summary.sort_values("qvalue", ascending=False).iloc[0]
summary_lines = [
    f"- Positive result: strongest descriptive agreement was {top_hit['label']} (Mann-Whitney U={top_hit['u_statistic']:.1f}, q={top_hit['qvalue']:.3g}, AUROC={top_hit['auroc']:.2f}).",
    f"- Negative result: weakest supported target here was {weakest['label']} (Mann-Whitney q={weakest['qvalue']:.3g}, AUROC={weakest['auroc']:.2f}).",
    "- Culture agreement is organism-group level, not strain-level or resistance-level agreement.",
]
display(Markdown("## Working Interpretation\n" + "\n".join(summary_lines)))
