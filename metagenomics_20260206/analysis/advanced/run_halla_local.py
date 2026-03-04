#!/usr/bin/env python3
"""Run HAllA locally with a minimal startup shim for missing optional R packages.

The installed HAllA package imports the R packages ``XICOR`` and ``eva`` at module
import time even when running a non-xicor, non-permutation workflow. This runner
shims those imports so the package can execute in ``spearman`` mode, which uses
analytic p-values and does not need either R package.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rpy2.robjects import packages as rpackages


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "inputs"
OUT_DIR = ROOT / "halla"
PREP_DIR = OUT_DIR / "inputs"
RESULT_DIR = OUT_DIR / "output"
SUMMARY_DIR = OUT_DIR / "tables"


class _MissingRPackage:
    def __init__(self, name: str) -> None:
        self.name = name

    def __getattr__(self, attr: str):
        raise RuntimeError(
            f'R package "{self.name}" is required for this HAllA code path but is not installed.'
        )


def patch_optional_r_imports() -> None:
    real_importr = rpackages.importr

    def patched_importr(name: str, *args, **kwargs):
        if name in {"XICOR", "eva"}:
            return _MissingRPackage(name)
        return real_importr(name, *args, **kwargs)

    rpackages.importr = patched_importr


def prepare_inputs() -> tuple[Path, Path]:
    PREP_DIR.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(INPUT_DIR / "halla_metadata_numeric.tsv", sep="\t")
    microbiome = pd.read_csv(
        INPUT_DIR / "halla_microbiome_samples_by_features.tsv", sep="\t", index_col=0
    )

    # HAllA is an association screen, not a repeated-measures model, so exclude
    # identifier-style columns that would otherwise dominate the signal.
    metadata = metadata.drop(columns=["patient_id", "visit_id"], errors="ignore")
    metadata = metadata.set_index("sample_id")
    metadata = metadata.replace({True: 1.0, False: 0.0})
    metadata = metadata.apply(pd.to_numeric, errors="raise").astype(float)

    shared_samples = [sample for sample in metadata.index if sample in microbiome.index]
    metadata = metadata.loc[shared_samples]
    microbiome = microbiome.loc[shared_samples]
    microbiome = microbiome.apply(pd.to_numeric, errors="raise").astype(float)

    metadata_t = metadata.transpose()
    metadata_t.index.name = "feature"
    metadata_path = PREP_DIR / "metadata_features_by_samples.tsv"
    metadata_t.to_csv(metadata_path, sep="\t")

    microbiome_t = microbiome.transpose()
    microbiome_t.index.name = "feature"
    microbiome_path = PREP_DIR / "microbiome_features_by_samples.tsv"
    microbiome_t.to_csv(microbiome_path, sep="\t")

    return metadata_path, microbiome_path


def summarize_results() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    all_assoc = pd.read_csv(RESULT_DIR / "all_associations.txt", sep="\t")
    sig_clusters = pd.read_csv(RESULT_DIR / "sig_clusters.txt", sep="\t")

    top_pairs = (
        all_assoc.sort_values(["q-values", "p-values", "association"], ascending=[True, True, False])
        .assign(abs_association=lambda df: df["association"].abs())
        .sort_values(["q-values", "abs_association"], ascending=[True, False])
        .drop(columns=["abs_association"])
    )
    top_pairs.head(100).to_csv(SUMMARY_DIR / "top_pairwise_associations.tsv", sep="\t", index=False)

    if not sig_clusters.empty:
        sig_clusters = sig_clusters.rename(
            columns={
                "cluster_X": "metadata_features",
                "cluster_Y": "microbiome_features",
            }
        )
        sig_clusters["metadata_size"] = sig_clusters["metadata_features"].str.count(";").fillna(0).astype(int) + 1
        sig_clusters["microbiome_size"] = sig_clusters["microbiome_features"].str.count(";").fillna(0).astype(int) + 1
    sig_clusters.to_csv(SUMMARY_DIR / "significant_clusters_summary.tsv", sep="\t", index=False)


def main() -> None:
    patch_optional_r_imports()
    metadata_path, microbiome_path = prepare_inputs()

    from halla import HAllA

    halla = HAllA(
        pdist_metric="spearman",
        out_dir=str(RESULT_DIR),
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
    summarize_results()


if __name__ == "__main__":
    main()
