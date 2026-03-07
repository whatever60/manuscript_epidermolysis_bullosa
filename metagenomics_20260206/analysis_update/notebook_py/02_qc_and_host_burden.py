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
# # 02. QC And Host Burden
#
# This notebook quantifies read-depth loss, host contamination, and the first batch-adjusted host-fraction regression used before mixed models.
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
# ## Rebuild The QC Table And Simple Host Model
#
# The QC outputs and first host-burden regression are generated here directly so the notebook shows the actual analysis path.
#

# %%
analysis_context = wc.base_analysis_context(context)
qc = base_data["qc"].copy()
host_fit, host_table = base.fit_host_model(qc)

qc_table = qc.reset_index().rename(columns={"index": "sample_id"})
wc.save_table(qc_table, wc.table_path(context, 2, "qc_metrics"))
wc.save_table(host_table, wc.table_path(context, 3, "host_model"))


# %% [markdown]
# ## Review Numbered Outputs
#

# %%
qc_table = pd.read_csv(wc.table_path(context, 2, "qc_metrics"), sep="\t")
host_table = pd.read_csv(wc.table_path(context, 3, "host_model"), sep="\t")

display(
    qc_table[
        [
            "sample_id",
            "patient_id",
            "visit_id",
            "host_removed_fraction",
            "bacterial_species_reads",
            "metaphlan_species_reads",
            "community_qc_pass",
        ]
    ].head(20)
)
display(host_table)

summary_lines = [
    f"- Median host fraction: {qc_table['host_removed_fraction'].median():.1%}.",
    f"- Community-analysis QC passing samples: {int(qc_table['community_qc_pass'].sum())} / {qc_table.shape[0]}.",
    f"- Very low-depth samples (<10,000 bacterial species reads): {int((~qc_table['community_qc_pass']).sum())}.",
    "- Positive result: the cluster-robust host model now adjusts for body site, chronicity, culture positivity, patient-relative elapsed time, and fixed batch-date effects.",
    "- Negative result: this notebook still uses a Gaussian working model for host fraction and handles repeated structure only through patient-clustered standard errors.",
]
display(Markdown("## Positive / Negative Takeaways\n" + "\n".join(summary_lines)))
