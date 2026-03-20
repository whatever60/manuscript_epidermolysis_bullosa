# analysis_concise

This folder is a self-contained, reduced workflow extracted from `analysis_update`.
It includes only notebooks/scripts needed to regenerate the figures and tables used in
`analysis_update/README.md` Results sections.

## Included Notebooks

Execution order:

1. `02_qc_and_host_burden.ipynb`
2. `03_bracken_community_structure.ipynb`
3. `04_culture_concordance.ipynb`
4. `06_mixed_models_and_repeated_measures.ipynb`
5. `12_adjusted_community_structure_analysis.ipynb` (R kernel)
6. `12_adjusted_community_structure_plots.ipynb`
7. `13_culture_threshold_and_concordance_analysis.ipynb` (R kernel)
8. `13_culture_threshold_and_concordance_plots.ipynb`
9. `14_host_fraction_gaussian_story_figures.ipynb`

Paired source files are in `notebook_py/` and synced with Jupytext.

## Core Shared Code

- `workflow_core.py`
- `analysis_base.py`
- `analysis_advanced.py`

No imports from `analysis_update` are used at runtime.

## README-Linked Outputs Regenerated Here

### Tables

- `tables/table_02_01_qc_metrics.tsv`
- `tables/table_06_01_patient_structure_diagnostics.tsv`
- `tables/table_06_02_host_mixed_effects.tsv`
- `tables/table_06_03_host_mixed_status.tsv`
- `tables/table_14_02_host_gaussian_random_effects.tsv`
- `tables/table_03_02_pairwise_distance_summary.tsv`
- `tables/table_12_02_pairwise_similarity_mixed_effects.tsv`
- `tables/table_12_04_pairwise_adjusted_margins.tsv`
- `tables/table_04_01_culture_concordance.tsv`
- `tables/table_13_01_culture_threshold_sweep.tsv`
- `tables/table_13_02_culture_optimal_thresholds.tsv`
- `tables/table_13_03_culture_mixed_concordance.tsv`
- `tables/table_13_04_culture_venn_counts.tsv`

### Figures

- `figures/fig_14_01_host_fraction_overview.svg`
- `figures/fig_14_02_host_gaussian_mixed_summary.svg`
- `figures/fig_14_03_host_gaussian_followup.svg`
- `figures/fig_14_04_host_gaussian_random_intercepts.svg`
- `figures/fig_03_01_pairwise_distance.svg`
- `figures/fig_12_02_pairwise_similarity_mixed.svg`
- `figures/fig_12_03_pairwise_adjusted_margins.svg`
- `figures/fig_13_04_culture_adjusted_concordance.svg`
- `figures/fig_13_03_culture_abundance_density.svg`

