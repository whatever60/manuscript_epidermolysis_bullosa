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
#     display_name: eb
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 15. Host-Fraction And Bacterial-Genus Barplots
#
# This notebook reuses the existing Bracken parsing/QC pipeline and reproduces
# patient-faceted barplot views:
# 1) host fraction per sample, defined as Homo sapiens reads / total Bracken species-level reads
# 2) bacterial-genus relative abundance per sample, with top-N genera retained
#    and all remaining genera collapsed into Others.
#

# %%
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import SVG, Markdown, display
from matplotlib.patches import Patch

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import workflow_core as wc

context, base_data, base, advanced = wc.bootstrap_notebook()


# %% [markdown]
# ## Inputs And Shared Helpers
#

# %%
qc = base_data["qc"].copy()
species_bac_counts = base_data["species_bac"].copy()

# Keep the shared sample universe for both host and bacterial-genus plots.
common_samples = qc.index.intersection(species_bac_counts.index)
qc = qc.loc[common_samples].copy()
species_bac_counts = species_bac_counts.loc[common_samples].copy()

N_TOP_GENUS = 20

FIG_HOST = context.figure_dir / "fig_15_01_host_fraction_by_patient.svg"
FIG_GENUS = context.figure_dir / "fig_15_02_bacterial_genus_by_patient.svg"


def save_svg_and_jpg(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".jpg"), bbox_inches="tight", dpi=300)


def patient_sort_key(series: pd.Series) -> pd.Series:
    as_num = pd.to_numeric(series, errors="coerce")
    if as_num.notna().all():
        return as_num.astype(int)
    return series.astype(str)


def sample_letter_sort(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


# %% [markdown]
# ## Build Host-Fraction Plot Data
#
# Host fraction here is explicitly defined as:
# Homo sapiens reads / total Bracken species-level reads.
#

# %%
qc["host_fraction_root"] = np.where(
    qc["bracken_root_reads"] > 0,
    qc["human_species_reads"] / qc["bracken_root_reads"],
    np.nan,
)

host_plot_df = (
    qc[
        [
            "patient_id",
            "sample_letter",
            "code",
            "body_region",
            "host_fraction_root",
            "bracken_root_reads",
            "human_species_reads",
        ]
    ]
    .dropna(subset=["host_fraction_root"])
    .copy()
)
host_plot_df["patient_id"] = host_plot_df["patient_id"].astype(str)
host_plot_df["sample_label"] = host_plot_df["sample_letter"].astype(str)
host_plot_df["body_region_label"] = host_plot_df["body_region"].map(
    base.BODY_REGION_LABELS
).fillna("Others")
host_plot_df = host_plot_df.sort_values(
    ["patient_id", "sample_letter", "code"], key=patient_sort_key
)


# %% [markdown]
# ## Figure 15.01: Host Fraction Barplot (One Axis Per Patient, Single Long Row)
#

# %%
patient_ids = host_plot_df["patient_id"].drop_duplicates().tolist()
n_patients = len(patient_ids)

body_regions_present = [
    key
    for key in base.BODY_REGION_ORDER
    if key in host_plot_df["body_region"].dropna().unique().tolist()
]
if "others" not in body_regions_present and "others" in host_plot_df["body_region"].tolist():
    body_regions_present.append("others")

region_palette = dict(
    zip(body_regions_present, sns.color_palette("Set2", n_colors=len(body_regions_present)))
)

host_width_ratios = []
for patient in patient_ids:
    n_samples = host_plot_df.loc[host_plot_df["patient_id"] == patient].shape[0]
    host_width_ratios.append(max(1, n_samples))

host_total_samples = int(sum(host_width_ratios))
host_fig_width = max(16.0, host_total_samples * 0.42 + n_patients * 0.35 + 4.5)

fig, axes = plt.subplots(
    nrows=1,
    ncols=n_patients,
    figsize=(host_fig_width, 4.2),
    sharey=True,
    gridspec_kw={"width_ratios": host_width_ratios, "wspace": 0.18},
)
axes = np.atleast_1d(axes).ravel()

for idx, patient in enumerate(patient_ids):
    ax = axes[idx]
    sub = host_plot_df.loc[host_plot_df["patient_id"] == patient].copy()
    sub = sub.sort_values(["sample_letter", "code"], key=sample_letter_sort)
    x = np.arange(sub.shape[0])
    colors = [region_palette.get(k, "#999999") for k in sub["body_region"].tolist()]

    ax.bar(x, sub["host_fraction_root"].to_numpy() * 100, color=colors, width=0.50)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["sample_label"].tolist(), fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_title(f"Patient {int(float(patient)):02d} (n={sub.shape[0]})", fontsize=10)
    if idx == 0:
        ax.set_ylabel("Host fraction (%)", fontsize=9)

legend_handles = [
    Patch(facecolor=region_palette[k], edgecolor="none", label=base.BODY_REGION_LABELS.get(k, k))
    for k in body_regions_present
]
fig.legend(
    handles=legend_handles,
    title="Body region",
    loc="center left",
    bbox_to_anchor=(1.01, 0.5),
    frameon=False,
)
fig.suptitle(
    "Host fraction by sample and patient\n(Homo sapiens reads / total Bracken species-level reads)",
    y=1.02,
    fontsize=13,
)
fig.tight_layout(rect=(0, 0, 0.86, 0.98))
save_svg_and_jpg(fig, FIG_HOST)
plt.close(fig)


# %% [markdown]
# ## Build Bacterial-Genus Relative-Abundance Plot Data
#
# Reuse the same top-N + Others strategy from the original Bracken visualization
# workflow, but at genus level within bacterial-domain counts.
#

# %%
species_rel = species_bac_counts.div(species_bac_counts.sum(axis=1), axis=0).fillna(0.0)

# Reuse Bracken taxonomy parser when possible; fallback to name-based genus parsing.
_, _, tax_df = base.parse_bracken_reports(str(context.data_dir / "kraken"))
genus_cols = [col for col in tax_df.columns if re.fullmatch(r"G\\d*", col)]


def fallback_genus(species_name: str) -> str:
    cleaned = species_name.replace("[", "").replace("]", "").strip()
    tokens = cleaned.split()
    if not tokens:
        return "Unknown"
    first = tokens[0].lower()
    if first in {"uncultured", "unclassified", "unidentified"} and len(tokens) >= 2:
        return tokens[1]
    if first == "candidatus" and len(tokens) >= 2:
        return f"Candidatus {tokens[1]}"
    return tokens[0]


def infer_genus(species_name: str) -> str:
    if species_name in tax_df.index and genus_cols:
        values = tax_df.loc[species_name, genus_cols]
        if isinstance(values, pd.DataFrame):
            values = values.iloc[0]
        for col in genus_cols:
            value = values[col]
            if pd.notna(value):
                return str(value)
    return fallback_genus(species_name)


species_to_genus = {species: infer_genus(species) for species in species_rel.columns}
genus_rel = species_rel.T.groupby(species_to_genus).sum().T

top_genera = (
    genus_rel.mean(axis=0).sort_values(ascending=False).head(N_TOP_GENUS).index.tolist()
)
minor_genera = [g for g in genus_rel.columns if g not in top_genera]

genus_plot = genus_rel[top_genera].copy()
if minor_genera:
    genus_plot["Others"] = genus_rel[minor_genera].sum(axis=1)
else:
    genus_plot["Others"] = 0.0

# Stack order requested: bottom -> top follows decreasing mean abundance,
# with Others forced to the top of the stack.
genus_mean = genus_plot.mean(axis=0)
non_other_desc = [
    genus
    for genus in genus_mean.sort_values(ascending=False).index.tolist()
    if genus != "Others"
]
stack_order = non_other_desc + ["Others"]
genus_plot = genus_plot[stack_order]

genus_plot = genus_plot.join(
    qc[["patient_id", "sample_letter", "code", "body_region"]],
    how="inner",
)
genus_plot["patient_id"] = genus_plot["patient_id"].astype(str)
genus_plot["sample_label"] = genus_plot["sample_letter"].astype(str)


# %% [markdown]
# ## Figure 15.02: Bacterial-Genus Stacked Barplot (One Axis Per Patient, Single Long Row)
#

# %%
patient_ids_genus = genus_plot["patient_id"].drop_duplicates().tolist()
n_patients_genus = len(patient_ids_genus)

genus_non_other = [g for g in stack_order if g != "Others"]
genus_palette_non_other = sns.color_palette(
    "tab20b", n_colors=max(1, len(genus_non_other))
)
genus_color_map = dict(zip(genus_non_other, genus_palette_non_other))
genus_color_map["Others"] = "#9e9e9e"

genus_width_ratios = []
for patient in patient_ids_genus:
    n_samples = genus_plot.loc[genus_plot["patient_id"] == patient].shape[0]
    genus_width_ratios.append(max(1, n_samples))

genus_total_samples = int(sum(genus_width_ratios))
genus_fig_width = max(16.0, genus_total_samples * 0.42 + n_patients_genus * 0.35 + 5.5)

fig, axes = plt.subplots(
    nrows=1,
    ncols=n_patients_genus,
    figsize=(genus_fig_width, 4.6),
    sharey=True,
    gridspec_kw={"width_ratios": genus_width_ratios, "wspace": 0.18},
)
axes = np.atleast_1d(axes).ravel()

for idx, patient in enumerate(patient_ids_genus):
    ax = axes[idx]
    sub = genus_plot.loc[genus_plot["patient_id"] == patient].copy()
    sub = sub.sort_values(["sample_letter", "code"], key=sample_letter_sort)

    x = np.arange(sub.shape[0])
    bottom = np.zeros(sub.shape[0], dtype=float)

    for genus in stack_order:
        values = sub[genus].to_numpy() * 100
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=genus_color_map[genus],
            width=0.50,
            linewidth=0,
            label=genus,
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(sub["sample_label"].tolist(), fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_title(f"Patient {int(float(patient)):02d} (n={sub.shape[0]})", fontsize=10)
    if idx == 0:
        ax.set_ylabel("Bacterial genus relative abundance (%)", fontsize=9)

legend_handles = [
    Patch(facecolor=genus_color_map[g], edgecolor="none", label=g) for g in stack_order
]
fig.legend(
    handles=legend_handles,
    title="Genus",
    loc="center left",
    bbox_to_anchor=(1.01, 0.5),
    frameon=False,
    ncol=1,
)
fig.suptitle(
    "Bacterial-genus composition by sample and patient\n(Top genera + Others, within bacterial-domain reads)",
    y=1.02,
    fontsize=13,
)
fig.tight_layout(rect=(0, 0, 0.84, 0.98))
save_svg_and_jpg(fig, FIG_GENUS)
plt.close(fig)


# %% [markdown]
# ## Review Outputs
#

# %%
display(SVG(filename=str(FIG_HOST)))
display(SVG(filename=str(FIG_GENUS)))

summary = pd.DataFrame(
    {
        "figure": [
            FIG_HOST.name,
            FIG_GENUS.name,
        ],
        "description": [
            "Host-fraction barplot by patient facets (Homo sapiens / root Bracken)",
            "Bacterial-genus stacked barplot by patient facets (top genera + Others)",
        ],
    }
)
display(summary)

top_preview = (
    genus_plot[stack_order]
    .drop(columns=["Others"], errors="ignore")
    .mean(axis=0)
    .sort_values(ascending=False)
    .head(15)
    .rename("mean_relative_abundance")
    .to_frame()
)
display(Markdown("### Top genera by mean relative abundance (for reference)"))
display(top_preview)
