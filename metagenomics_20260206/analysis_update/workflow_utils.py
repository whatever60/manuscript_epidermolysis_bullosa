#!/usr/bin/env python3
from __future__ import annotations

"""Compatibility wrapper for older analysis_update imports.

New notebook code should import `workflow_core` directly.
"""

from workflow_core import (
    WorkflowContext,
    advanced_analysis_context,
    base_analysis_context,
    bootstrap_notebook,
    ensure_dirs,
    extract_metaphlan_species_matrix,
    figure_path,
    get_context,
    load_modules,
    prepare_base_data,
    save_table,
    set_plot_defaults,
    table_path,
)

__all__ = [
    "WorkflowContext",
    "advanced_analysis_context",
    "base_analysis_context",
    "bootstrap_notebook",
    "ensure_dirs",
    "extract_metaphlan_species_matrix",
    "figure_path",
    "get_context",
    "load_modules",
    "prepare_base_data",
    "save_table",
    "set_plot_defaults",
    "table_path",
]
