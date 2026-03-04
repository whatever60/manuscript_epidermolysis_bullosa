#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@dataclass
class WorkflowContext:
    data_dir: Path
    output_dir: Path
    figure_dir: Path
    table_dir: Path
    halla_dir: Path


TABLE_ID_MAP = {
    1: (1, 1),
    2: (2, 1),
    3: (2, 2),
    4: (3, 1),
    5: (3, 2),
    6: (4, 1),
    7: (5, 1),
    8: (6, 1),
    9: (6, 2),
    10: (6, 3),
    11: (6, 4),
    12: (6, 5),
    13: (6, 6),
    14: (7, 1),
    15: (7, 2),
    16: (7, 3),
    17: (7, 4),
    18: (7, 5),
    19: (7, 6),
    20: (8, 1),
    21: (8, 2),
    22: (8, 3),
    23: (9, 1),
    24: (9, 2),
    25: (10, 1),
    26: (10, 2),
    27: (11, 1),
    28: (11, 2),
    29: (12, 1),
    30: (12, 2),
    31: (12, 3),
    32: (13, 1),
    33: (13, 2),
    34: (13, 3),
    35: (13, 4),
    36: (14, 1),
    37: (14, 2),
    38: (12, 4),
}

FIGURE_ID_MAP = {
    1: (2, 1),
    2: (3, 1),
    3: (4, 1),
    4: (5, 1),
    5: (6, 1),
    6: (6, 2),
    7: (7, 1),
    8: (8, 1),
    9: (9, 1),
    10: (10, 1),
    11: (11, 1),
    12: (12, 1),
    13: (12, 2),
    14: (13, 1),
    15: (13, 2),
    16: (13, 3),
    17: (13, 4),
    18: (14, 1),
    19: (14, 2),
    20: (14, 3),
    21: (14, 4),
    22: (12, 3),
}


def get_context() -> WorkflowContext:
    output_dir = Path(__file__).resolve().parent
    data_dir = output_dir.parent
    return WorkflowContext(
        data_dir=data_dir,
        output_dir=output_dir,
        figure_dir=output_dir / "figures",
        table_dir=output_dir / "tables",
        halla_dir=output_dir / "halla_output",
    )


def ensure_dirs(context: WorkflowContext) -> None:
    context.output_dir.mkdir(exist_ok=True)
    context.figure_dir.mkdir(exist_ok=True)
    context.table_dir.mkdir(exist_ok=True)
    context.halla_dir.mkdir(exist_ok=True)


def set_plot_defaults() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_modules():
    workflow_dir = get_context().output_dir
    base = _load_module("analysis_update_base_kernel", workflow_dir / "analysis_base.py")
    advanced = _load_module("analysis_update_advanced_kernel", workflow_dir / "analysis_advanced.py")
    return base, advanced


def base_analysis_context(context: WorkflowContext):
    base, _ = load_modules()
    return base.AnalysisContext(
        data_dir=context.data_dir,
        analysis_dir=context.output_dir,
        figure_dir=context.figure_dir,
        table_dir=context.table_dir,
    )


def advanced_analysis_context(context: WorkflowContext):
    _, advanced = load_modules()
    return advanced.AdvancedContext(
        data_dir=context.data_dir,
        output_dir=context.output_dir,
        figure_dir=context.figure_dir,
        table_dir=context.table_dir,
        input_dir=context.output_dir / "tool_inputs",
    )


def table_path(context: WorkflowContext, table_number: int, slug: str) -> Path:
    notebook_number, order = TABLE_ID_MAP[table_number]
    return context.table_dir / f"table_{notebook_number:02d}_{order:02d}_{slug}.tsv"


def figure_path(context: WorkflowContext, figure_number: int, slug: str) -> Path:
    notebook_number, order = FIGURE_ID_MAP[figure_number]
    return context.figure_dir / f"fig_{notebook_number:02d}_{order:02d}_{slug}.svg"


def save_table(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, sep="\t", index=False)
    return path


def extract_metaphlan_species_matrix(metaphlan: pd.DataFrame) -> pd.DataFrame:
    species_cols = [col for col in metaphlan.columns if "|s__" in col and "|t__" not in col]
    species = metaphlan[species_cols].copy()
    species.columns = [
        col.split("|s__", 1)[1].replace("_", " ").strip()
        for col in species.columns
    ]
    species = species.T.groupby(level=0).sum().T
    return species.loc[:, species.sum(axis=0) > 0]


def prepare_base_data(context: WorkflowContext, base=None) -> dict[str, pd.DataFrame]:
    if base is None:
        base, _ = load_modules()
    analysis_context = base_analysis_context(context)
    species_all, species_bac = base.load_bracken_tables(analysis_context)
    metaphlan = base.load_metaphlan_table(analysis_context)
    sample_ids = sorted(species_all.index)
    metadata = base.load_metadata(analysis_context, sample_ids)
    qc = base.prepare_qc_table(analysis_context, metadata, species_all, species_bac, metaphlan)
    qc["patient_id"] = qc["patient_id"].astype(str)
    qc["visit_id"] = qc["visit_id"].astype(str)
    return {
        "metadata": metadata,
        "qc": qc,
        "species_all": species_all,
        "species_bac": species_bac,
        "metaphlan": metaphlan,
        "metaphlan_species": extract_metaphlan_species_matrix(metaphlan),
    }


def bootstrap_notebook() -> tuple[WorkflowContext, dict[str, pd.DataFrame], object, object]:
    context = get_context()
    ensure_dirs(context)
    set_plot_defaults()
    base, advanced = load_modules()
    base_data = prepare_base_data(context, base=base)
    return context, base_data, base, advanced
