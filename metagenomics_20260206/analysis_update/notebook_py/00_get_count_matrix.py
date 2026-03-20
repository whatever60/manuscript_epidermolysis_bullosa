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
# # 00. Get Canonical Count Matrices
#
# This notebook regenerates and saves the three canonical count matrices used across analysis notebooks:
# Bracken species-all, Bracken bacteria-only species, and MetaPhlAn clade read counts.
# It also verifies these regenerated tables are exactly equal (after sample-ID normalization)
# to the legacy top-level matrices previously consumed by the workflow.
#

# %%
import os
import sys
from glob import glob
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import workflow_core as wc

context = wc.get_context()
wc.ensure_dirs(context)
base, _ = wc.load_modules()


# %% [markdown]
# ## Regenerate Bracken Species Matrices From `kraken_with_host`
#
# This follows the same parser logic used in the original preprocessing notebook:
# species-all keeps all species-level taxa, and species-bacteria keeps only species-level taxa under domain Bacteria.
#

# %%
abundance_df, read_count_df, tax_df = base.parse_bracken_reports(
    str(context.data_dir / "kraken_with_host")
)

abundance_df.index = abundance_df.index.map(base.rename_sample)
read_count_df.index = read_count_df.index.map(base.rename_sample)
abundance_df = abundance_df.sort_index()
read_count_df = read_count_df.sort_index()

tax2level = {
    idx: tax_df.columns[np.where(~row.isnull())[0].max()] for idx, row in tax_df.iterrows()
}

read_count_species_all = read_count_df[
    [taxon for taxon in abundance_df.columns if tax2level[taxon] == "S"]
].astype(int)

read_count_species_bac = read_count_df[
    [
        taxon
        for taxon in abundance_df.columns
        if tax2level[taxon] == "S" and tax_df.loc[taxon, "D"] == "Bacteria"
    ]
].astype(int)

print(read_count_species_all.shape, read_count_species_bac.shape)


# %% [markdown]
# ## Regenerate MetaPhlAn Count Matrix From Raw MetaPhlAn Output
#
# For each sample, clade relative abundance is converted back to read counts using the
# `# ... reads processed` header value, then merged across samples.
#

# %%
def load_metaphlan_matrix(pattern: str) -> pd.DataFrame:
    file_paths = sorted(glob(pattern))
    sample_series = []

    for path in file_paths:
        sample_name = os.path.splitext(os.path.basename(path))[0]
        reads_processed = -1
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("#"):
                    if line.endswith(" reads processed\n"):
                        reads_processed = int(line.split()[0][1:])
                else:
                    break
            content = line + handle.read()

        table = pd.read_csv(
            StringIO(content),
            sep="\t",
            comment="#",
            header=None,
            names=["clade_name", "ncbi_tax_id", sample_name, "additional_species"],
        )
        sample_series.append(table.set_index("clade_name")[sample_name] * reads_processed)

    merged = pd.concat(sample_series, axis=1).fillna(0)
    merged = merged.round().astype(int)
    return merged.transpose()


read_count_metaphlan = load_metaphlan_matrix(str(context.data_dir / "metaphlan" / "*.txt"))
read_count_metaphlan.index = read_count_metaphlan.index.map(base.rename_sample)
read_count_metaphlan = read_count_metaphlan.groupby(level=0).sum().sort_index()

print(read_count_metaphlan.shape)


# %% [markdown]
# ## Save Canonical Matrices Into `analysis_update/tables`
#

# %%
species_all_out = context.table_dir / "table_00_01_read_count_species_all.csv"
species_bac_out = context.table_dir / "table_00_02_read_count_species_bac.csv"
metaphlan_out = context.table_dir / "table_00_03_read_count_metaphlan.tsv"

read_count_species_all.to_csv(species_all_out)
read_count_species_bac.to_csv(species_bac_out)
read_count_metaphlan.to_csv(metaphlan_out, sep="\t")

print(species_all_out)
print(species_bac_out)
print(metaphlan_out)


# %% [markdown]
# ## Confirm Equivalence To Legacy Matrices Used Previously
#
# Equality is checked after sample-ID normalization and duplicate-sample collapsing, matching downstream notebook behavior.
#

# %%
def load_count_matrix(path: Path, sep: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=sep, index_col=0)
    frame.index = frame.index.map(base.normalize_sample_id)
    frame = frame.groupby(level=0).sum().sort_index()
    return frame


def compare_matrices(
    matrix_name: str,
    generated: pd.DataFrame,
    legacy_path: Path,
    sep: str,
) -> dict[str, object]:
    if not legacy_path.exists():
        return {
            "matrix": matrix_name,
            "legacy_exists": False,
            "n_samples_generated": int(generated.shape[0]),
            "n_features_generated": int(generated.shape[1]),
            "n_samples_legacy": np.nan,
            "n_features_legacy": np.nan,
            "index_equal": False,
            "columns_equal": False,
            "max_abs_diff": np.nan,
            "exact_equal": False,
            "generated_file": generated.name,
            "legacy_file": str(legacy_path),
        }

    legacy = load_count_matrix(legacy_path, sep=sep)
    generated_norm = generated.copy()
    generated_norm.index = generated_norm.index.map(base.normalize_sample_id)
    generated_norm = generated_norm.groupby(level=0).sum().sort_index()

    aligned_generated = generated_norm.reindex(
        index=legacy.index, columns=legacy.columns, fill_value=0
    )
    max_abs_diff = float((aligned_generated - legacy).abs().to_numpy().max())

    return {
        "matrix": matrix_name,
        "legacy_exists": True,
        "n_samples_generated": int(generated_norm.shape[0]),
        "n_features_generated": int(generated_norm.shape[1]),
        "n_samples_legacy": int(legacy.shape[0]),
        "n_features_legacy": int(legacy.shape[1]),
        "index_equal": bool(generated_norm.index.equals(legacy.index)),
        "columns_equal": bool(generated_norm.columns.equals(legacy.columns)),
        "max_abs_diff": max_abs_diff,
        "exact_equal": bool(generated_norm.equals(legacy)),
        "generated_file": generated.name,
        "legacy_file": str(legacy_path),
    }


read_count_species_all.name = str(species_all_out)
read_count_species_bac.name = str(species_bac_out)
read_count_metaphlan.name = str(metaphlan_out)

matrix_checks = pd.DataFrame(
    [
        compare_matrices(
            "read_count_species_all",
            read_count_species_all,
            context.data_dir / "read_count_species_all.csv",
            sep=",",
        ),
        compare_matrices(
            "read_count_species_bac",
            read_count_species_bac,
            context.data_dir / "read_count_species_bac.csv",
            sep=",",
        ),
        compare_matrices(
            "read_count_metaphlan",
            read_count_metaphlan,
            context.data_dir / "read_count_metaphlan.tsv",
            sep="\t",
        ),
    ]
)

check_out = context.table_dir / "table_00_04_count_matrix_equivalence.tsv"
matrix_checks.to_csv(check_out, sep="\t", index=False)
matrix_checks


# %%
if not matrix_checks["exact_equal"].all():
    raise RuntimeError(
        "At least one regenerated matrix did not exactly match the legacy matrix."
    )

print(f"All regenerated matrices are exactly equal to legacy inputs. Check table: {check_out}")
