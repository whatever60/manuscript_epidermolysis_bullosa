# Advanced mixed-effects layer

## Summary

- These models add variance components for patient and visit on top of the original fixed-effect analyses.
- Patient-only mixed models were often singular in this cohort, so the working specification uses variance components for both patient and visit with a constant grouping factor.
- Sign agreement between cluster-robust and mixed-model estimates was 73.1% across terms that were estimable in both frameworks.

## Host models

- Base host mixed model: no fixed-effect term reached q <= 0.1; the largest signal was `C(collection_year)[T.2024]` with estimate 4.22 (q=0.356).
- Extended host mixed model: chronicity signal `Acute-like vs unknown` had estimate 2.02 (q=0.356).
- Host base model variance components: patient 1.29e-09, visit 1.64.

## Species models

- Staphylococcus aureus: Per log10 bacterial reads -> 1.40 (95% CI 0.69 to 2.11, q=0.00562).
- Pseudomonas aeruginosa: Per log10 bacterial reads -> 1.14 (95% CI 0.53 to 1.75, q=0.00562).
- Pseudomonas aeruginosa: Chronic-like vs unknown -> 2.86 (95% CI 1.28 to 4.44, q=0.00626).
- Klebsiella pneumoniae: Per log10 bacterial reads -> 1.27 (95% CI 0.43 to 2.10, q=0.033).
- Serratia marcescens: Per log10 bacterial reads -> 1.06 (95% CI 0.35 to 1.76, q=0.033).
- Staphylococcus aureus: Head / neck vs lower extremity -> 2.71 (95% CI 0.77 to 4.64, q=0.0491).

## Method assessment

- statsmodels_mixedlm: ran. Python variance-component mixed models executed successfully.
- HAllA: blocked_dependency. Installed CLI fails to start in the current environment. Detail: Missing R package XICOR required by the installed HAllA entrypoint.
- MaAsLin2: missing. R package is not installed locally. Inputs were prepared because this is a strong follow-up multivariable method for this dataset.
- LEfSe: missing. Executable/package is not installed locally. Inputs were prepared, but the method is less suitable here because it does not model repeated measures and multiple covariates as directly as mixed models or MaAsLin2.

## Notes

- MaAsLin2 remains the best external follow-up for multivariable microbiome association once the R package is installed in the `eb` environment.
- LEfSe was considered but kept as a prepared-input option only, because it is fundamentally less aligned with repeated-measures multi-covariate inference.
- HAllA is potentially useful for exploratory microbe-metadata block associations, but the local installation is blocked by a missing R dependency (`XICOR`).

## Figures

1. `advanced_figure_01_host_compare.svg`: Host-fraction coefficient comparison between the original cluster-robust regression and the advanced variance-component mixed model.
2. `advanced_figure_02_species_mixed.svg`: Mixed-model fixed effects for selected taxa, emphasizing body-region and chronicity terms.

## Tables

- `host_mixed_effects.tsv` and `host_mixed_status.tsv` summarize host mixed models.
- `species_mixed_effects.tsv` and `species_mixed_status.tsv` summarize taxon mixed models.
- `mixed_vs_cluster_comparison.tsv` compares advanced-model coefficients against the earlier cluster-robust estimates.
- `method_status.tsv` records which optional tools ran and which were blocked by missing dependencies.

## Species shown in Figure 2

- Corynebacterium striatum: Acute-like vs unknown -> -2.65 (95% CI -4.76 to -0.53).
- Cutibacterium acnes: Head / neck vs lower extremity -> 1.97 (95% CI 0.42 to 3.51).
- Staphylococcus aureus: Head / neck vs lower extremity -> 2.71 (95% CI 0.77 to 4.64).
- Pseudomonas aeruginosa: Chronic-like vs unknown -> 2.86 (95% CI 1.28 to 4.44).
- Klebsiella pneumoniae: Mixed vs unknown -> 3.80 (95% CI 0.61 to 6.99).
