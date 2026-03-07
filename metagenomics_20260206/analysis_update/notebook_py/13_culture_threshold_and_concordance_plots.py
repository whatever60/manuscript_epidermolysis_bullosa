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
# # 13. Culture Threshold And Concordance Plots (Python)
#
# This notebook reads the numbered table outputs from the R analysis notebook
# and generates Figure 13 panels directly in Python/matplotlib/seaborn.
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
# ## Load Table Inputs
#

# %%
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import SVG
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

try:
    from matplotlib_venn import venn2
except Exception:
    venn2 = None

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

threshold_sweep = pd.read_csv(
    wc.table_path(context, 32, "culture_threshold_sweep"), sep="\t"
)
optimal_thresholds = pd.read_csv(
    wc.table_path(context, 33, "culture_optimal_thresholds"), sep="\t"
)
mixed_concordance = pd.read_csv(
    wc.table_path(context, 34, "culture_mixed_concordance"), sep="\t"
)
venn_counts = pd.read_csv(wc.table_path(context, 35, "culture_venn_counts"), sep="\t")
descriptive_concordance = pd.read_csv(
    wc.table_path(context, 42, "culture_concordance_descriptive"), sep="\t"
)
abundance_plot_df = pd.read_csv(
    wc.table_path(context, 43, "culture_abundance_plot_data"), sep="\t"
)


# %% [markdown]
# ## Define Plotting Helpers
#

# %%
DISPLAY_THRESHOLDS = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]


def save_svg_and_jpg(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".jpg"), bbox_inches="tight", dpi=300)


def axes_grid(
    n_panels: int,
    ncols: int = 3,
    panel_width: float = 4.5,
    panel_height: float = 3.4,
    sharex: bool = False,
    sharey: bool = False,
):
    ncols = max(1, min(ncols, n_panels if n_panels > 0 else 1))
    nrows = max(1, math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_width * ncols, panel_height * nrows),
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
    )
    flat_axes = [ax for row in axes for ax in row]
    return fig, flat_axes


def plot_threshold_sweep(
    threshold_data: pd.DataFrame,
    optimal_data: pd.DataFrame,
    output_path: Path,
) -> None:
    merged = threshold_data.merge(
        optimal_data[["group", "threshold"]].rename(
            columns={"threshold": "threshold_opt"}
        ),
        on="group",
        how="left",
    )
    label_order = (
        threshold_data[["group", "label"]]
        .drop_duplicates()
        .sort_values("group")["label"]
        .tolist()
    )
    fig, axes = axes_grid(
        len(label_order),
        ncols=3,
        panel_width=4.4,
        panel_height=3.1,
    )
    for idx, label in enumerate(label_order):
        ax = axes[idx]
        sub = merged.loc[merged["label"] == label].sort_values("threshold")
        ax.plot(sub["threshold"] * 100.0, sub["f1"], color="#3b6a8f", linewidth=1.4)
        if sub["threshold_opt"].notna().any():
            x_opt = float(sub["threshold_opt"].dropna().iloc[0] * 100.0)
            ax.axvline(x_opt, color="#b22222", linestyle="--", linewidth=1.0)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Threshold (%)", fontsize=9)
        ax.set_ylabel("F1", fontsize=9)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1.05)
        ax.tick_params(labelsize=8)

    for ax in axes[len(label_order) :]:
        ax.set_axis_off()

    fig.suptitle("Culture-versus-metagenomics threshold sweep", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_svg_and_jpg(fig, output_path)
    plt.close(fig)


def circle_intersection_area(r1: float, r2: float, d: float) -> float:
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    term1 = r1**2 * math.acos((d**2 + r1**2 - r2**2) / (2 * d * r1))
    term2 = r2**2 * math.acos((d**2 + r2**2 - r1**2) / (2 * d * r2))
    term3 = 0.5 * math.sqrt(
        (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)
    )
    return term1 + term2 - term3


def solve_center_distance(r1: float, r2: float, target_intersection: float) -> float:
    max_overlap = math.pi * min(r1, r2) ** 2
    target = min(max(target_intersection, 0.0), max_overlap)
    if target <= 0:
        return r1 + r2 + max(r1, r2) * 0.35
    if abs(target - max_overlap) < 1e-12:
        return abs(r1 - r2)
    lo = abs(r1 - r2) + 1e-10
    hi = r1 + r2 - 1e-10
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        area = circle_intersection_area(r1, r2, mid)
        if area > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def style_venn_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_with_custom_solver(
    ax: plt.Axes,
    culture_only: float,
    sequencing_only: float,
    both: float,
    neither: float,
) -> None:
    a_size = max(culture_only + both, 0.0)
    b_size = max(sequencing_only + both, 0.0)
    overlap = max(both, 0.0)

    if a_size <= 0 and b_size <= 0:
        ax.text(
            0.5,
            0.57,
            "No positive calls",
            ha="center",
            va="center",
            fontsize=8,
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.35,
            f"Neither: {int(round(neither))}",
            ha="center",
            va="center",
            fontsize=8,
            transform=ax.transAxes,
        )
        style_venn_axis(ax)
        return

    r1 = max(math.sqrt(a_size / math.pi), 1e-8)
    r2 = max(math.sqrt(b_size / math.pi), 1e-8)
    distance = solve_center_distance(r1, r2, overlap)

    c1, c2 = 0.0, distance
    x_min = min(c1 - r1, c2 - r2)
    x_max = max(c1 + r1, c2 + r2)
    y_lim = max(r1, r2)
    pad = max(0.22 * max(r1, r2), 0.2)
    center_shift = -0.5 * (x_min + x_max)
    c1 += center_shift
    c2 += center_shift

    ax.add_patch(
        Circle(
            (c1, 0),
            r1,
            facecolor="#3b6a8f",
            edgecolor="#2f4d63",
            alpha=0.35,
            linewidth=1.0,
        )
    )
    ax.add_patch(
        Circle(
            (c2, 0),
            r2,
            facecolor="#b22222",
            edgecolor="#7f1b1b",
            alpha=0.35,
            linewidth=1.0,
        )
    )
    ax.text(
        c1 - 0.45 * r1,
        0,
        f"{int(round(culture_only))}",
        ha="center",
        va="center",
        fontsize=8,
    )
    ax.text(
        (c1 + c2) / 2,
        0,
        f"{int(round(overlap))}",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
    )
    ax.text(
        c2 + 0.45 * r2,
        0,
        f"{int(round(sequencing_only))}",
        ha="center",
        va="center",
        fontsize=8,
    )
    ax.text(
        0.5 * (x_min + x_max) + center_shift,
        -(y_lim + 0.55 * pad),
        f"Neither: {int(round(neither))}",
        ha="center",
        va="center",
        fontsize=7,
    )
    ax.set_xlim(x_min - pad + center_shift, x_max + pad + center_shift)
    ax.set_ylim(-(y_lim + 1.0 * pad), y_lim + 0.9 * pad)
    style_venn_axis(ax)


def draw_proportional_venn(
    ax: plt.Axes,
    culture_only: float,
    sequencing_only: float,
    both: float,
    neither: float,
) -> None:
    if venn2 is not None:
        venn = venn2(
            subsets=(culture_only, sequencing_only, both),
            set_labels=("", ""),
            ax=ax,
        )
        if venn is not None:
            style_map = {"10": "#3b6a8f", "01": "#b22222", "11": "#6c4a86"}
            for key, color in style_map.items():
                patch = venn.get_patch_by_id(key)
                if patch is not None:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.35)
                    patch.set_edgecolor("#404040")
                    patch.set_linewidth(0.9)

            label_values = {
                "10": int(round(culture_only)),
                "11": int(round(both)),
                "01": int(round(sequencing_only)),
            }
            for key, value in label_values.items():
                text = venn.get_label_by_id(key)
                if text is not None:
                    text.set_text(str(value))
                    text.set_fontsize(8)
                    if key == "11":
                        text.set_fontweight("bold")

            ax.text(
                0.5,
                0.03,
                f"Neither: {int(round(neither))}",
                ha="center",
                va="bottom",
                transform=ax.transAxes,
                fontsize=7,
            )
            style_venn_axis(ax)
            return

    draw_with_custom_solver(ax, culture_only, sequencing_only, both, neither)


def plot_venn_grid(venn_data: pd.DataFrame, output_path: Path) -> None:
    venn_df = venn_data.copy()
    venn_df["threshold"] = venn_df["threshold"].astype(float)
    available = sorted(venn_df["threshold"].unique().tolist())
    cutoff_levels = [
        value
        for value in DISPLAY_THRESHOLDS
        if any(math.isclose(value, a, rel_tol=0, abs_tol=1e-9) for a in available)
    ]
    if not cutoff_levels:
        cutoff_levels = available

    group_df = venn_df[["group", "label"]].drop_duplicates()
    group_order = group_df["group"].tolist()
    group_to_label = dict(zip(group_df["group"], group_df["label"]))

    n_rows = len(group_order)
    n_cols = len(cutoff_levels)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.45 * n_cols, 2.05 * n_rows),
        squeeze=False,
    )

    for row_idx, group in enumerate(group_order):
        for col_idx, cutoff in enumerate(cutoff_levels):
            ax = axes[row_idx, col_idx]
            row = venn_df.loc[
                (venn_df["group"] == group)
                & (
                    np.isclose(
                        venn_df["threshold"],
                        cutoff,
                        rtol=0.0,
                        atol=1e-9,
                    )
                )
            ]
            if row.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    fontsize=8,
                    transform=ax.transAxes,
                )
                style_venn_axis(ax)
            else:
                rec = row.iloc[0]
                draw_proportional_venn(
                    ax=ax,
                    culture_only=float(rec["culture_only"]),
                    sequencing_only=float(rec["sequencing_only"]),
                    both=float(rec["both"]),
                    neither=float(rec["neither"]),
                )

            if row_idx == 0:
                ax.set_title(f"{cutoff * 100:.1f}%", fontsize=9, pad=4)
            if col_idx == 0:
                ax.text(
                    -0.26,
                    0.5,
                    group_to_label[group],
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    fig.suptitle(
        "Culture-only / Both / Sequencing-only overlaps (area-proportional)",
        y=0.997,
        fontsize=12,
    )
    fig.text(
        0.5,
        0.01,
        "Display cutoff for metagenomic relative abundance",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.03, 0.03, 1.0, 0.985))
    save_svg_and_jpg(fig, output_path)
    plt.close(fig)


def plot_abundance_boxplot(
    abundance_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    q_lookup = {}
    if "label" in summary_df.columns and "qvalue" in summary_df.columns:
        q_lookup = summary_df.set_index("label")["qvalue"].to_dict()

    label_order = (
        abundance_df[["group", "label"]]
        .drop_duplicates()
        .sort_values("group")["label"]
        .tolist()
    )
    status_order = ["Culture negative", "Culture positive"]
    palette = {
        "Culture negative": "#4c78a8",
        "Culture positive": "#d65f5f",
    }

    fig, axes = axes_grid(
        len(label_order),
        ncols=3,
        panel_width=2.0,
        panel_height=3.0,
    )
    for idx, label in enumerate(label_order):
        ax = axes[idx]
        sub = abundance_df.loc[abundance_df["label"] == label].copy()
        sns.boxplot(
            data=sub,
            x="culture_status",
            y="log10_rel_abundance",
            order=status_order,
            palette=palette,
            width=0.62,
            fliersize=3.0,
            linewidth=1.0,
            ax=ax,
        )
        qvalue = q_lookup.get(label, np.nan)
        if pd.notna(qvalue):
            ax.set_title(f"{label} U-test q={qvalue:.3f}", fontsize=10)
        else:
            ax.set_title(label, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("log10(rel. abundance + 1e-6)", fontsize=9)
        ax.set_xticklabels(
            [tick.get_text().split()[-1].capitalize() for tick in ax.get_xticklabels()],
            rotation=20,
            ha="right",
            fontsize=8,
        )
        ax.set_yticklabels(
            [tick.get_text() for tick in ax.get_yticklabels()],
            fontsize=8,
        )

    for ax in axes[len(label_order) :]:
        ax.set_axis_off()

    fig.suptitle(
        "Metagenomic abundance by culture status (boxplots)",
        y=0.995,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_svg_and_jpg(fig, output_path)
    plt.close(fig)


def plot_adjusted_concordance(
    concordance_df: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_df = concordance_df.loc[
        concordance_df["status"].isin(["ok", "ok_singular"])
    ].copy()
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(
            0.5,
            0.5,
            "No adjusted concordance models converged.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        save_svg_and_jpg(fig, output_path)
        plt.close(fig)
        return

    plot_df["significant"] = plot_df["qvalue"].fillna(1.0) <= 0.1
    plot_df["fit_quality"] = np.where(
        plot_df["status"] == "ok_singular",
        "Singular fit",
        "Regular fit",
    )
    plot_df = plot_df.sort_values("estimate").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 0.55 * len(plot_df)))
    ax.axvline(0, color="#808080", linestyle="--", linewidth=0.9)
    color_map = {True: "#b22222", False: "#3b6a8f"}
    marker_map = {"Regular fit": "o", "Singular fit": "^"}

    y_values = np.arange(len(plot_df))
    for idx, row in plot_df.iterrows():
        x = float(row["estimate"])
        low = float(row["conf.low"])
        high = float(row["conf.high"])
        color = color_map[bool(row["significant"])]
        marker = marker_map[row["fit_quality"]]
        ax.errorbar(
            x=x,
            y=idx,
            xerr=[[x - low], [high - x]],
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.1,
            capsize=3,
            markersize=7,
        )

    ax.set_yticks(y_values)
    ax.set_yticklabels(plot_df["label"].tolist())
    ax.set_xlabel(
        "Adjusted difference in rank-normalized abundance\n(Culture positive - Culture negative)"
    )
    ax.set_ylabel("")
    ax.set_title("Technical/nuisance-adjusted rank-based culture concordance models")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#b22222",
            markeredgecolor="#b22222",
            label="q <= 0.1",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#3b6a8f",
            markeredgecolor="#3b6a8f",
            label="q > 0.1",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            linestyle="None",
            label="Regular fit",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="black",
            linestyle="None",
            label="Singular fit",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(1, 0),
    )
    fig.tight_layout()
    save_svg_and_jpg(fig, output_path)
    plt.close(fig)


# %% [markdown]
# ## Render Figure 13 Panels From Numbered Tables
#

# %%
fig_threshold = wc.figure_path(context, 14, "culture_threshold_sweep")
plot_threshold_sweep(threshold_sweep, optimal_thresholds, fig_threshold)

fig_venn = wc.figure_path(context, 15, "culture_venn_diagrams")
plot_venn_grid(venn_counts, fig_venn)

fig_abundance = wc.figure_path(context, 16, "culture_abundance_density")
plot_abundance_boxplot(abundance_plot_df, descriptive_concordance, fig_abundance)

fig_adjusted = wc.figure_path(context, 17, "culture_adjusted_concordance")
plot_adjusted_concordance(mixed_concordance, fig_adjusted)

display(SVG(filename=str(fig_threshold)))
display(SVG(filename=str(fig_venn)))
display(SVG(filename=str(fig_abundance)))
display(SVG(filename=str(fig_adjusted)))


# %%
fig_threshold = wc.figure_path(context, 14, "culture_threshold_sweep")
plot_threshold_sweep(threshold_sweep, optimal_thresholds, fig_threshold)

fig_venn = wc.figure_path(context, 15, "culture_venn_diagrams")
plot_venn_grid(venn_counts, fig_venn)

fig_abundance = wc.figure_path(context, 16, "culture_abundance_density")
plot_abundance_boxplot(abundance_plot_df, descriptive_concordance, fig_abundance)

fig_adjusted = wc.figure_path(context, 17, "culture_adjusted_concordance")
plot_adjusted_concordance(mixed_concordance, fig_adjusted)

display(SVG(filename=str(fig_threshold)))
display(SVG(filename=str(fig_venn)))
display(SVG(filename=str(fig_abundance)))
display(SVG(filename=str(fig_adjusted)))


# %%
