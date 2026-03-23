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
# 1) species-level classified-read composition per sample, split into bacteria domain, human,
#    and non-bacterial/non-human residual
# 2) bacterial-genus relative abundance per sample, removing missing-genus assignments,
#    renormalizing, retaining genera with >=10% in at least one sample, and collapsing the rest into Others.
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

GENUS_KEEP_THRESHOLD = 0.10

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


def patient_facet_width_ratios(sample_counts: list[int]) -> list[float]:
    return [max(1.0, float(count)) for count in sample_counts]


def set_patient_facet_xlim(ax: plt.Axes, n_bars: int) -> None:
    x_span = max(1.0, float(n_bars))
    ax.set_xlim(-0.5, -0.5 + x_span)
    ax.margins(x=0.0)


def separated_tab20b(n_colors: int) -> list[tuple[float, float, float]]:
    base_palette = list(sns.color_palette("tab20b", n_colors=20))
    reordered: list[tuple[float, float, float]] = []
    for offset in range(4):
        reordered.extend(base_palette[offset::4])
    return reordered[:n_colors]


# %% [markdown]
# ## Build Species-Level Composition Plot Data
#
# This first barplot uses three mutually exclusive fractions from the Bracken species-all table:
# bacteria domain, Homo sapiens (shown in the legend as Human), and the non-bacterial/non-human residual.
#

# %%
composition_plot_df = (
    qc[
        [
            "patient_id",
            "sample_letter",
            "code",
            "human_species_fraction",
            "bacterial_species_fraction",
            "non_bacterial_non_human_fraction",
        ]
    ]
    .dropna(
        subset=[
            "human_species_fraction",
            "bacterial_species_fraction",
            "non_bacterial_non_human_fraction",
        ]
    )
    .copy()
)
composition_plot_df["patient_id"] = composition_plot_df["patient_id"].astype(str)
composition_plot_df["sample_label"] = composition_plot_df["sample_letter"].astype(str)
composition_plot_df = composition_plot_df.sort_values(
    ["patient_id", "sample_letter", "code"], key=patient_sort_key
)
composition_plot_df["Bacteria domain"] = composition_plot_df["bacterial_species_fraction"]
composition_plot_df["Human"] = composition_plot_df["human_species_fraction"]
composition_plot_df["Others"] = composition_plot_df["non_bacterial_non_human_fraction"]


# %% [markdown]
# ## Figure 15.01: Species-Level Composition Barplot (One Axis Per Patient, Single Long Row)
#

# %%
patient_ids = composition_plot_df["patient_id"].drop_duplicates().tolist()
n_patients = len(patient_ids)

composition_order = ["Human", "Bacteria domain", "Others"]
composition_palette = {
    "Human": "#b75d69",
    "Bacteria domain": "#3f7f4c",
    "Others": "#8d99ae",
}

host_sample_counts = [
    composition_plot_df.loc[composition_plot_df["patient_id"] == patient].shape[0]
    for patient in patient_ids
]
host_width_ratios = patient_facet_width_ratios(host_sample_counts)

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
    sub = composition_plot_df.loc[composition_plot_df["patient_id"] == patient].copy()
    sub = sub.sort_values(["sample_letter", "code"], key=sample_letter_sort)
    x = np.arange(sub.shape[0])
    bottom = np.zeros(sub.shape[0], dtype=float)
    for category in composition_order:
        values = sub[category].to_numpy() * 100
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=composition_palette[category],
            width=0.50,
            linewidth=0,
            label=category,
        )
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(sub["sample_label"].tolist(), fontsize=8)
    set_patient_facet_xlim(ax, sub.shape[0])
    ax.set_ylim(0, 100)
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_title(f"Patient {int(float(patient)):02d} (n={sub.shape[0]})", fontsize=10)
    if idx == 0:
        ax.set_ylabel("Classified-read fraction (%)", fontsize=9)

legend_handles = [
    Patch(facecolor=composition_palette[k], edgecolor="none", label=k)
    for k in composition_order
]
fig.legend(
    handles=legend_handles,
    title="Category",
    loc="center left",
    bbox_to_anchor=(1.01, 0.5),
    frameon=False,
)
fig.suptitle(
    "Species-level classified-read composition by sample and patient\n(Bacteria domain, Human, and non-bacterial/non-human residual)",
    y=1.02,
    fontsize=13,
)
fig.tight_layout(rect=(0, 0, 0.86, 0.98))
save_svg_and_jpg(fig, FIG_HOST)
plt.close(fig)


# %% [markdown]
# ## Build Bacterial-Genus Relative-Abundance Plot Data
#
# Apply the manuscript genus-filtering rule at bacterial-domain genus level:
# remove missing-genus assignments, renormalize, keep genera with max >= 10% in any sample, and collapse the rest into Others.
#

# %%
species_rel = species_bac_counts.div(species_bac_counts.sum(axis=1), axis=0).fillna(0.0)

# Reuse Bracken taxonomy parser when possible; fallback to name-based genus parsing.
_, _, tax_df = base.parse_bracken_reports(str(context.data_dir / "kraken"))
genus_cols = [col for col in tax_df.columns if re.fullmatch(r"G\\d*", col)]


def fallback_genus(species_name: str) -> str | float:
    cleaned = species_name.replace("[", "").replace("]", "").strip()
    tokens = cleaned.split()
    if not tokens:
        return np.nan
    first = tokens[0].lower()
    # Treat non-genus placeholders as missing genus labels.
    if first in {
        "uncultured",
        "unclassified",
        "unidentified",
        "endosymbiotic",
        "endosymbiont",
        "bacterium",
    }:
        return np.nan
    if first == "candidatus":
        if len(tokens) >= 2:
            return f"Candidatus {tokens[1]}"
        return np.nan
    return tokens[0]


def infer_genus(species_name: str) -> str | float:
    if species_name in tax_df.index and genus_cols:
        values = tax_df.loc[species_name, genus_cols]
        if isinstance(values, pd.DataFrame):
            values = values.iloc[0]
        for col in genus_cols:
            value = values[col]
            if pd.notna(value) and str(value).strip() != "":
                return str(value)
    return fallback_genus(species_name)


species_to_genus = pd.Series({species: infer_genus(species) for species in species_rel.columns})
missing_genus_species = species_to_genus[species_to_genus.isna()].index.tolist()

if missing_genus_species:
    missing_genus_fraction_by_sample = species_rel[missing_genus_species].sum(axis=1)
else:
    missing_genus_fraction_by_sample = pd.Series(0.0, index=species_rel.index)

missing_genus_mean_fraction = float(missing_genus_fraction_by_sample.mean())

species_rel_labeled = species_rel.drop(columns=missing_genus_species, errors="ignore").copy()
renorm_denom = species_rel_labeled.sum(axis=1)
species_rel_labeled = species_rel_labeled.div(renorm_denom.replace(0, np.nan), axis=0).fillna(0.0)

species_to_genus_labeled = species_to_genus.dropna().astype(str)
genus_rel = species_rel_labeled.T.groupby(species_to_genus_labeled).sum().T

keep_genera = genus_rel.columns[genus_rel.max(axis=0) >= GENUS_KEEP_THRESHOLD].tolist()
minor_genera = [g for g in genus_rel.columns if g not in keep_genera]

genus_plot = genus_rel[keep_genera].copy() if keep_genera else pd.DataFrame(index=genus_rel.index)
genus_plot["Others"] = genus_rel[minor_genera].sum(axis=1) if minor_genera else 0.0

# Stack order: decreasing mean abundance (bottom to top), Others on top.
genus_mean = genus_plot.mean(axis=0)
non_other_desc = [g for g in genus_mean.sort_values(ascending=False).index.tolist() if g != "Others"]
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
genus_palette_non_other = separated_tab20b(max(1, len(genus_non_other)))
genus_color_map = dict(zip(genus_non_other, genus_palette_non_other))
genus_color_map["Others"] = "#9e9e9e"

genus_sample_counts = [
    genus_plot.loc[genus_plot["patient_id"] == patient].shape[0]
    for patient in patient_ids_genus
]
genus_width_ratios = patient_facet_width_ratios(genus_sample_counts)

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
    set_patient_facet_xlim(ax, sub.shape[0])
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
    "Bacterial-genus composition by sample and patient\n(Genera with >=10% in >=1 sample + Others, after removing missing-genus assignments)",
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
            "Species-level classified-read composition by patient facets (Bacteria domain, Human, Others)",
            "Bacterial-genus stacked barplot by patient facets (>=10% in >=1 sample + Others)",
        ],
    }
)
display(summary)

top_preview = (
    genus_plot[stack_order]
    .drop(columns=["Others"], errors="ignore")
    .mean(axis=0)
    .sort_values(ascending=False)
    .rename("mean_relative_abundance")
    .to_frame()
)

filter_summary = pd.DataFrame(
    {
        "metric": [
            "Missing-genus species removed (count)",
            "Average relative abundance removed before renormalization",
            "Genera retained by max >= 0.10 rule",
        ],
        "value": [
            len(missing_genus_species),
            missing_genus_mean_fraction,
            len(non_other_desc),
        ],
    }
)

display(Markdown("### Genus-filtering summary"))
display(filter_summary)

display(Markdown("### Retained genera by mean relative abundance (for reference)"))
display(top_preview)
