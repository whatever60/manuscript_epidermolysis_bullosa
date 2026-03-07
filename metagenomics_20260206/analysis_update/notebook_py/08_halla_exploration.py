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
# # 08. HAllA Exploration
#
# This notebook runs HAllA as an exploratory metadata-microbiome block-association screen.
# It is intentionally separated from the main inferential notebooks because HAllA does not explicitly model repeated measures.
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
# ## Define Notebook-Local HAllA Helpers
#
# These helper functions are kept in this notebook because they are only used here.
#

# %%
import shutil

from rpy2.robjects import packages as rpackages


EMPTY_TOP_PAIRS = pd.DataFrame(
    columns=["X_features", "Y_features", "association", "p-values", "q-values"]
)
EMPTY_SIG_CLUSTERS = pd.DataFrame(
    columns=[
        "cluster_rank",
        "metadata_features",
        "microbiome_features",
        "best_adjusted_pvalue",
        "metadata_size",
        "microbiome_size",
    ]
)


class _MissingRPackage:
    def __init__(self, name):
        self.name = name

    def __getattr__(self, attr):
        raise RuntimeError(
            f'R package "{self.name}" is required for this HAllA code path but is not installed.'
        )


def patch_optional_r_imports():
    real_importr = rpackages.importr

    def patched_importr(name, *args, **kwargs):
        if name in {"XICOR", "eva"}:
            return _MissingRPackage(name)
        return real_importr(name, *args, **kwargs)

    rpackages.importr = patched_importr


def import_halla_class():
    try:
        from halla import HAllA

        return HAllA
    except Exception as primary_exc:
        fallback_site = Path("/home/ubuntu/miniforge3/lib/python3.12/site-packages")
        if fallback_site.exists() and str(fallback_site) not in sys.path:
            sys.path.append(str(fallback_site))
        # Clear any failed or shadowed module entry before retrying import.
        sys.modules.pop("halla", None)
        try:
            from halla import HAllA

            return HAllA
        except Exception as secondary_exc:
            raise ImportError(
                "Unable to import HAllA from current env or base Miniforge site-packages. "
                f"primary={primary_exc!r}; fallback={secondary_exc!r}"
            )


def prepare_halla_inputs():
    input_dir = context.halla_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    qc = base_data["qc"]
    species_bac = base_data["species_bac"]

    sample_ids = qc.index[qc["model_qc_pass"]].tolist()
    rel_ab = species_bac.loc[sample_ids].copy()
    rel_ab = rel_ab.div(rel_ab.sum(axis=1), axis=0).fillna(0)
    prevalence = (rel_ab > 0).mean(axis=0)
    rel_ab = rel_ab.loc[:, prevalence >= 0.1]

    metadata = qc.loc[
        sample_ids,
        [
            "host_removed_fraction",
            "log10_bacterial_reads",
            "culture_s_aureus",
            "culture_p_aeruginosa",
            "culture_serratia_marcescens",
            "culture_proteus_mirabilis",
            "culture_gas",
            "culture_klebsiella_spp",
            "culture_e_coli",
            "culture_acinetobacter_baumannii",
            "culture_e_faecalis",
            "body_region",
            "chronicity_group",
            "clinical_infection_flag",
        ],
    ].copy()
    metadata = pd.get_dummies(
        metadata,
        columns=["body_region", "chronicity_group", "clinical_infection_flag"],
        dtype=float,
    )
    metadata = metadata.replace({True: 1.0, False: 0.0})
    metadata = metadata.apply(pd.to_numeric, errors="raise").astype(float)

    metadata_t = metadata.transpose()
    metadata_t.index.name = "feature"
    metadata_path = input_dir / "metadata_features_by_samples.tsv"
    metadata_t.to_csv(metadata_path, sep="\t")

    microbiome_t = rel_ab.transpose()
    microbiome_t.index.name = "feature"
    microbiome_path = input_dir / "microbiome_features_by_samples.tsv"
    microbiome_t.to_csv(microbiome_path, sep="\t")
    return metadata_path, microbiome_path


def summarize_halla_results(output_dir):
    all_assoc = pd.read_csv(output_dir / "all_associations.txt", sep="\t")
    sig_clusters = pd.read_csv(output_dir / "sig_clusters.txt", sep="\t")

    top_pairs = (
        all_assoc.assign(abs_association=lambda df: df["association"].abs())
        .sort_values(["q-values", "abs_association"], ascending=[True, False])
        .drop(columns=["abs_association"])
        .head(100)
        .reset_index(drop=True)
    )

    sig_clusters = sig_clusters.rename(
        columns={
            "cluster_X": "metadata_features",
            "cluster_Y": "microbiome_features",
            "best_adjusted_pvalue": "best_adjusted_pvalue",
        }
    )
    sig_clusters["metadata_size"] = (
        sig_clusters["metadata_features"].str.count(";").fillna(0).astype(int) + 1
    )
    sig_clusters["microbiome_size"] = (
        sig_clusters["microbiome_features"].str.count(";").fillna(0).astype(int) + 1
    )
    return top_pairs, sig_clusters


# %% [markdown]
# ## Run HAllA And Save Numbered Outputs
#

# %%
patch_optional_r_imports()
metadata_path, microbiome_path = prepare_halla_inputs()
figure_file = wc.figure_path(context, 8, "halla_top25")
if figure_file.exists():
    figure_file.unlink()

output_dir = context.halla_dir / "output"
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

rows = []
top_pairs = EMPTY_TOP_PAIRS.copy()
sig_clusters = EMPTY_SIG_CLUSTERS.copy()

try:
    HAllA = import_halla_class()

    halla = HAllA(
        pdist_metric="spearman",
        out_dir=str(output_dir),
        verbose=True,
        num_threads=1,
        force_permutations=False,
    )
    halla.load(str(metadata_path), str(microbiome_path))
    halla.run()
    halla.generate_hallagram(
        block_num=25,
        x_dataset_label="Clinical features",
        y_dataset_label="Bacterial species",
        output_file="hallagram_top25",
        plot_type="svg",
    )
    top_pairs, sig_clusters = summarize_halla_results(output_dir)
    rows.append(
        {
            "method": "HAllA",
            "status": "ran_with_local_shim",
            "runnable": True,
            "reason": "Executed in spearman mode with optional R-package imports shimmed locally.",
            "detail": "Optional HAllA startup imports for XICOR/eva were shimmed so the non-xicor, non-permutation workflow could run.",
        }
    )
except Exception as exc:
    rows.append(
        {
            "method": "HAllA",
            "status": "failed",
            "runnable": False,
            "reason": "Execution failed in the local environment.",
            "detail": repr(exc),
        }
    )

method_status = pd.DataFrame(rows)
hallagram = output_dir / "hallagram_top25.svg"
if hallagram.exists():
    shutil.copy2(hallagram, figure_file)

wc.save_table(method_status, wc.table_path(context, 20, "halla_method_status"))
wc.save_table(top_pairs, wc.table_path(context, 21, "halla_top_pairwise_associations"))
wc.save_table(sig_clusters, wc.table_path(context, 22, "halla_significant_clusters"))


# %% [markdown]
# ## Review Numbered Outputs
#

# %%
method_status = pd.read_csv(wc.table_path(context, 20, "halla_method_status"), sep="\t")
top_pairs = pd.read_csv(
    wc.table_path(context, 21, "halla_top_pairwise_associations"), sep="\t"
)
sig_clusters = pd.read_csv(
    wc.table_path(context, 22, "halla_significant_clusters"), sep="\t"
)

display(method_status)
figure_file = wc.figure_path(context, 8, "halla_top25")
if figure_file.exists():
    display(SVG(filename=str(figure_file)))
display(top_pairs.head(20))
display(sig_clusters.head(20))

status = method_status.iloc[0]["status"]
summary_lines = [
    f"- HAllA status: {status}.",
    "- Positive result: HAllA provides an exploratory block-level view of metadata-microbiome associations.",
    "- Negative result: it is not the primary inferential layer here because it does not model patient and culture-date batch structure the way the mixed models do.",
]
display(Markdown("## Working Interpretation\n" + "\n".join(summary_lines)))
