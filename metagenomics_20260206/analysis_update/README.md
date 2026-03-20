## Contents (with hyperlinks)

- [Methods](#methods)
- [Modeling of host genomic DNA fraction](#modeling-of-host-genomic-dna-fraction)
- [Similarity analysis of shotgun metagenomic sequencing](#similarity-analysis-of-shotgun-metagenomic-sequencing)
- [Comparison between shotgun metagenomic sequencing and culture results](#comparison-between-shotgun-metagenomic-sequencing-and-culture-results)
- [Results](#results)
- [Host genomic DNA content in wound skin swabs is associated with structured batch effects and acute-like wound status](#host-genomic-dna-content-in-wound-skin-swabs-is-associated-with-structured-batch-effects-and-acute-like-wound-status)
- [Patient identity and wound chronicity remain associated with metagenomic similarity after technical adjustment](#patient-identity-and-wound-chronicity-remain-associated-with-metagenomic-similarity-after-technical-adjustment)
- [Shotgun metagenomic detection agrees with culture in a nonparametric descriptive analysis and remains significant after rank-based technical adjustment](#shotgun-metagenomic-detection-agrees-with-culture-in-a-nonparametric-descriptive-analysis-and-remains-significant-after-rank-based-technical-adjustment)
- [Third party packages used (including both python and R packages)](#third-party-packaged-used-including-both-python-and-r-packages)
- [Reproducibility](#reproducibility)

## Methods

### Definition of body region

To facilitate statistical analysis, we aggregated specific wound-location annotations into broader body regions and used body region as a key covariate in host-fraction modeling, metagenomic similarity analysis, and culture-concordance analysis.
Body region was derived from wound location in the following manner: head/neck (ear, earlobe, eye, nose, neck, forehead, antihelix), upper extremity (shoulder, arm, axilla, elbow, hand, finger, forearm), trunk/perineum (back, chest, abdomen, buttock, buttocks, groin, perineum, trunk), and lower extremity (knee, shin, ankle, foot, thigh, leg).
If multiple region keywords were present in one entry, assignment followed a fixed precedence order (head/neck, then upper extremity, then trunk/perineum, then lower extremity), and entries with no matched keyword were labeled others.
Assignments to the others category occurred for sample 01J (right posterior thight), sample 10F (right heel), and sample 10I (left antecubital fossa).
One sample (12F, left thigh/buttock) had wound location spanning more than one predefined body region (lower extremity and trunk/perineum), which was assigned to trunk/perineum by the fixed precedence rule.
No sequenced sample included in the final analysis set was assigned to the others body-region category.

### Modeling of host genomic DNA fraction

#### Host fraction definition

Taking the advantage that the Kraken standard database contains human (*Homo sapiens*), we applied Kraken/Bracken to the pre-host-filter reads for coherent quantifications of both human and bacterial species. The fraction of reads assigned to *Homo sapiens* was defined from the species-all count matrix as:

$$
h_i \;=\; \frac{\text{Homo\_sapiens\_reads}_i}{\sum_{s \in \text{species-all}} \text{reads}_{is}}.
$$

Under this species-total denominator, the classified non-bacterial/non-human residual (primarily fungi, archaea, viral, and other non-human eukaryotic assignments) was usually small, with median 0.0029% and 95th percentile 1.7%; therefore, this quantity was also approximately equivalent to human / (human + bacterial) for most samples in this cohort.
To obtain an approximately unbounded response for Gaussian modeling, host fraction was logit-transformed after truncation away from 0 and 1:
$$
y_i \;=\; \log\!\left(\frac{\tilde h_i}{1-\tilde h_i}\right),
\qquad
\tilde h_i = \min\{\max(h_i,10^{-4}),1-10^{-4}\}.
$$

#### Metadata encoding

All 74 sequenced samples were retained for host-fraction modeling after metadata cleaning and harmonization.
Culture date was treated as a technical batch variable (batch identifier).
Patient-relative biological time was defined as years since the first sampled date for that patient,
$$
t_i = \frac{\text{culture\_date}_i - \min(\text{culture\_date for patient }p_i)}{365.25}.
$$
Categorical variables with incomplete clinical annotation were not numerically imputed; instead, uncertainty was retained as explicit levels such as unknown.

#### Primary Gaussian mixed model

The main host-fraction model was a Gaussian linear mixed model:

$$
y_i
=
\beta_0
+
\sum_{r} \beta_r\,I(\text{body\_region}_i=r)
+
\sum_{c} \gamma_c\,I(\text{chronicity}_i=c)
+
\delta\,I(\text{culture\_positive}_i=\text{positive})
+
\eta\, t_i
+
u_{p(i)}
+
b_{k(i)}
+
\varepsilon_i.
$$

In this formulation, $r$ indexes the non-reference body-region levels (head/neck, upper extremity, and trunk/perineum, with lower extremity being the reference level), and $c$ indexes the non-reference chronicity levels (acute-like, chronic-like, and mixed, with unknown being the reference level).
The random components were $u_{p(i)} \sim N(0,\sigma^2_{\text{patient}})$ for patient-specific intercepts, $b_{k(i)} \sim N(0,\sigma^2_{\text{batch}})$ for culture-date batch intercepts, and $\varepsilon_i \sim N(0,\sigma^2)$ for residual error.
Patient and batch were modeled as random effects because they define correlation structure among observations rather than scientific contrasts of interest.
Body region, chronicity, broad culture positivity, and patient-relative elapsed time were modeled as fixed effects because the goal was to estimate interpretable cohort-level associations for those variables.

#### Model fitting and inference

Models were fit by maximum likelihood (reml = FALSE) so that likelihood-ratio tests between nested models were valid.
To reduce dependence on a single optimizer, each model was attempted with lbfgs, powell, and bfgs. All models successfully converged (except for the patient-only random effect model with lbfgs which failed due to singularity) and the lowest-AIC fit was retained.

Coefficient-level fixed-effect inference was based on Wald tests from the fitted Gaussian mixed model, with $z_j = \hat\beta_j / \operatorname{SE}(\hat\beta_j)$.
Two-sided p values were taken from the asymptotic normal approximation implemented in statsmodels.
Global Benjamini-Hochberg false discovery rate (BH-FDR) procedure was used for multiple-testing adjustment.

Significance thresholds were defined as $p < 0.05$ for nominal single-term tests and $q < 0.10$ for FDR-adjusted results.

Random-effect terms were assessed by nested likelihood-ratio tests comparing models with and without the relevant variance component.
For a nested comparison, $\Lambda = 2\{\ell(\text{full}) - \ell(\text{reduced})\}$.
Because the null hypothesis for a variance component is on the boundary of the parameter space ($H_0:\sigma^2 = 0$), p values were computed using the standard one-parameter boundary correction $p_{\text{boundary}} = \tfrac{1}{2}P(\chi^2_1 \ge \Lambda)$.

To test whether an entire categorical factor contributed to host fraction, we used likelihood-ratio omnibus tests comparing the full model to reduced models with the factor removed.
For chronicity group, the null hypothesis was $H_0:\gamma_{\text{acute}}=\gamma_{\text{chronic}}=\gamma_{\text{mixed}}=0$.
For body region, the null hypothesis was $H_0:\beta_{\text{head/neck}}=\beta_{\text{upper extremity}}=\beta_{\text{trunk/perineum}}=0$.
P values were obtained from a $\chi^2_d$ distribution with $d$ equal to the number of removed coefficients.

### Similarity analysis of shotgun metagenomic sequencing

#### Outcome definition and distance metric

To assess whether samples sharing biological covariates were more similar in shotgun metagenomic composition, we analyzed Bracken-derived bacterial community profiles from QC-passing samples (at least 10,000 bacterial Bracken species reads per sample) and quantified between-sample compositional dissimilarity using Bray-Curtis distance.
If $p_{is}$ denotes the relative abundance of species $s$ in sample $i$, Bray-Curtis distance between samples $i$ and $j$ was defined as
$$
d_{ij}=\frac{\sum_s |p_{is}-p_{js}|}{\sum_s (p_{is}+p_{js})}.
$$
Distances range from 0 for identical communities to 1 for maximally dissimilar communities.
All community-level analyses in this section were restricted to the QC-passing sample set used for composition-sensitive inference.

#### Descriptive pairwise comparisons

As an initial descriptive analysis, all sample pairs were partitioned into five descriptive groups: same patient/same site/close date (<6 months apart), same patient/close date, same patient/same site, same patient/different date-site, and different patient.
Median Bray-Curtis distances for the four within-patient groups were compared against the different-patient reference group using one-sided Mann-Whitney tests with the alternative hypothesis that within-group distances were smaller.
Resulting p values were adjusted with BH-FDR.

#### Pairwise mixed-effects similarity models

Because the biological question is inherently pairwise, we also modeled Bray-Curtis distance directly at the sample-pair level.
The pairwise dataset included all unordered pairs of QC-passing samples.
For each pair $(i,j)$, we defined binary indicators for whether the two samples shared the same patient, batch date, broad body region, chronicity group, or culture-positivity status.
We also defined continuous pairwise nuisance covariates as the mean host fraction, $\overline{h}_{ij}=(h_i+h_j)/2$, the mean bacterial read depth, $\overline{z}_{ij}=\{\log_{10}(\text{bacterial reads}_i)+\log_{10}(\text{bacterial reads}_j)\}/2$, and the absolute elapsed-time difference, $\Delta t_{ij}=|t_i-t_j|$,
where $t_i$ is years since the first sample from the same patient.

The primary linear mixed-effects model represented site similarity at the level of broad anatomical region:
$$
d_{ij}
=
\beta_0
+
\beta_1 I(\text{same patient}_{ij})
+
\beta_2 I(\text{same batch}_{ij})
+
\beta_3 I(\text{same body region}_{ij})
+
\beta_4 I(\text{same chronicity}_{ij})
+
\beta_5 I(\text{same culture positivity}_{ij})
+
\beta_6 \Delta t_{ij}
+
\beta_7 \overline{h}_{ij}
+
\beta_8 \overline{z}_{ij}
+
a_i + a_j + \varepsilon_{ij}.
$$
Here, $a_i$ and $a_j$ are crossed random intercepts for the two samples contributing to the pair, and $\varepsilon_{ij}$ is residual error.
This crossed random-effects structure accounts for non-independence arising because each sample contributes to many pairwise distances.

#### Fixed-effect inference and interpretation

For the pairwise mixed models, coefficient-level inference was based on Wald tests.
For each coefficient $\beta_k$, the test statistic was $t_k = \hat\beta_k / \operatorname{SE}(\hat\beta_k)$, with two-sided p values obtained from the fitted mixed model.
P values across coefficients within each pairwise model family were adjusted using BH-FDR to generate q values.
Significance for FDR-controlled inference was interpreted at $q < 0.10$, while nominal p values were also reported.


### Comparison between shotgun metagenomic sequencing and culture results

#### Organism-group abundance definition

To compare culture results with metagenomic profiling, nine clinically relevant organism groups were evaluated: *S. aureus*, *P. aeruginosa*, Serratia, Proteus, group A Streptococcus (GAS), Klebsiella spp., *E. coli*, *A. baumannii*, and *E. faecalis*.

Culture positivity for each group was assigned as follows: *S. aureus* was defined by the presence of methicillin-sensitive Staphylococcus aureus, methicillin-resistant Staphylococcus aureus, or Staphylococcus aureus; *P. aeruginosa* by Pseudomonas aeruginosa; Serratia by Serratia marcescens; Proteus by Proteus mirabilis; GAS by GAS, Streptococcus pyogenes, or Strep pyogenes; Klebsiella spp. by Klebsiella; *E. coli* by Escherichia coli or E. coli; *A. baumannii* by Acinetobacter baumannii; and *E. faecalis* by Enterococcus faecalis.

The corresponding metagenomic features were mapped at the species level as follows: Staphylococcus aureus for *S. aureus*, the sum of Pseudomonas aeruginosa and Pseudomonas sp. B111 for *P. aeruginosa*, Serratia marcescens for Serratia, Proteus mirabilis for Proteus, Streptococcus pyogenes for GAS, Escherichia coli for *E. coli*, Acinetobacter baumannii for *A. baumannii*, and Enterococcus faecalis for *E. faecalis*. For Klebsiella spp., the metagenomic abundance was defined as the sum of Klebsiella pneumoniae and Klebsiella oxytoca.

For each sample $i$ and organism group $g$, the metagenomic abundance was defined as the sum of the relative abundances of all species assigned to that group: $A_i^{(g)} = \sum_{s \in g} p_{is}$.

#### Nonparametric concordance

The primary descriptive concordance analysis compared organism-group abundance between culture-positive and culture-negative samples using a one-sided Mann-Whitney U test with alternative hypothesis culture positive > culture negative.
For each organism group, if $A_i$ denotes the relative abundance in sample $i$, the null hypothesis was

$$
H_0: A_i^{(+)} \text{ and } A_i^{(-)} \text{ come from the same distribution},
$$

against the one-sided alternative that abundances were stochastically larger in culture-positive samples.
As a complementary discrimination summary, AUROC was computed from empirical ROC curves using organism-group abundance as the continuous score and culture positivity as the binary label.
BH-FDR was applied across organism groups to obtain q values.
This descriptive analysis used all 74 sequenced samples.

#### Threshold sweep and overlap summaries

To characterize practical detection agreement across abundance cutoffs, we evaluated thresholds from 0 to 0.10 relative abundance in increments of 0.001.
At each threshold, sequencing calls were defined as positive if $A_i^{(g)}\ge \tau$.
For each organism group and threshold, we computed true positives, false positives, false negatives, and true negatives, then derived sensitivity, specificity, positive predictive value, negative predictive value, F1 score, and Cohen’s kappa.
The threshold with the highest F1 score, breaking ties by kappa and then by smaller threshold, was retained as the optimal threshold for each organism group.

#### Adjusted rank-based mixed-effects concordance model

To adjust for technical nuisance structure while remaining nonparametric with respect to abundance scale, we fit a rank-based linear mixed model for each organism group.
These models were restricted to samples with at least 5,000 bacterial Bracken reads.
For each organism-group abundance $A_i^{(g)}$, we computed its within-dataset rank $r_i = \operatorname{rank}(A_i^{(g)})$, using average ranks for ties, and then transformed ranks to a normal-score scale $z_i = \Phi^{-1}\left(\frac{r_i - 0.5}{n}\right)$, where $\Phi^{-1}$ is the inverse standard normal cumulative distribution function and $n$ is the number of samples in the filtered dataset.
The adjusted model was

$$z_i = \beta_0 + \beta_1 I(\text{culture positive}_i) +
\beta_2 \,\text{host fraction}_i
+
\beta_3 \log_{10}(\text{bacterial reads}_i)
+
u_{\text{patient}(i)}
+
b_{\text{batch}(i)}
+
\varepsilon_i,
$$

with $u_{\text{patient}(i)} \sim N(0,\sigma^2_{\text{patient}}), \, b_{\text{batch}(i)} \sim N(0,\sigma^2_{\text{batch}}), \, \varepsilon_i \sim N(0,\sigma^2)$.

Here, culture positive was the fixed effect of interest, while host fraction and log10 bacterial reads were nuisance fixed effects.
Patient and batch were modeled as random intercepts to account for repeated sampling within patient and technical batch structure associated with culture date.
In this updated force-all-target analysis, we attempted one adjusted model for each of the nine culture organism groups regardless of class size, and only marked a target as non-estimable when there was no class variation or no variation in the rank-normalized abundance response.
Fixed-effect p values for culture status were taken from Wald tests and adjusted across all nine organism groups by BH-FDR.
Singular mixed-model fits were explicitly flagged as boundary random-effect solutions and interpreted as unstable sensitivity-level evidence rather than robust primary inference.

## Results

### Host genomic DNA content in wound skin swabs is associated with patient and wound chronicity site.

The metagenomic sequencing also revealed the presence of bacterial genera other than *Pseudomonas* containing species capable of producing flagella, including *Serratia*, *Escherichia*, and *Citrobacter*.

Skin swabs are known to contain heterogenous and substantial host genomic DNA.
In our shotgun metagenomic sequencing, host fraction was high and variable across samples (range 0.0030 to 1.0; median 0.95) (Fig. 1B, Fig. S1B, S1C).
Rather than focusing entirely on bacterial reads, we used this fraction as a complementary signal of wound microenvironment and host carryover. We modeled logit-transformed host fraction in a Gaussian mixed model with patient and culture-date as random effects and body region, chronicity group, culture positivity, and years since first patient sample as fixed effects. Both random effects contributed materially to model fit: relative to a fixed-effects-only model, adding patient improved fit (LRT = 7.1, p = 0.0038) and adding culture-date improved fit (LRT = 3.4, p = 0.032) respectively; when added sequentially, patient remained informative on top of culture-date (LRT = 5.3, p = 0.011), whereas the additional culture-date contribution on top of patient was weaker (LRT = 1.6, p = 0.10), indicating stronger patient-level than incremental culture-date contribution in the fully adjusted setting. For fixed effects, no term remained significant after global BH adjustment, although acute-like chronicity had the lowest q value (q = 0.24), followed by upper-extremity body region (q = 0.33) (Fig. 1C, 1D). Culture positivity, elapsed time since first visit, and the remaining chronicity/location categories were weaker (q >= 0.89). The fitted random intercepts showed broad spread across both culture dates and patients (Fig. S1D, S1E), reinforcing that variation from these sources remained large even after fixed-effect adjustment. These findings support the use of host fraction as a complementary quantitative signal in wound metagenomic profiling.

> Tables:
> - table_00_01_read_count_species_all.csv (canonical Bracken species-all counts)
> - table_00_02_read_count_species_bac.csv (canonical Bracken bacterial species counts)
> - table_00_03_read_count_metaphlan.tsv (canonical MetaPhlAn clade counts)
> - table_00_04_count_matrix_equivalence.tsv (equivalence checks vs legacy top-level matrices)
> - table_02_01_qc_metrics.tsv (host ratio)
> - table_06_01_patient_structure_diagnostics.tsv (optimizer selection)
> - table_06_02_host_mixed_effects.tsv (beta and Wald p)
> - table_06_03_host_mixed_status.tsv (LTR and p)
> - table_14_02_host_gaussian_random_effects.tsv (random intercepts)
>
> Figures:
> - fig_15_02_bacterial_genus_by_patient.svg -> Fig. 1A
> - fig_14_01_host_fraction_overview.svg left -> Fig. 1B
> - fig_14_02_host_gaussian_mixed_summary.svg left -> Fig. 1D
> - fig_15_01_host_fraction_by_patient.svg -> Fig. S1B
> - fig_14_02_host_gaussian_mixed_summary.svg middle & right -> Fig. S1C & Fig. S1D
> - fig_14_03_host_gaussian_followup.svg -> Fig. 1C
> - fig_14_04_host_gaussian_random_intercepts.svg -> Fig. S1E & Fig. S1F
>


#### Figure captions

Figure 1. Host genomic DNA burden in EB wound swabs is structured by patient-level heterogeneity and wound-level biology. A) Bacterial-genus composition by sample and patient, shown as stacked relative-abundance bars (top genera plus Others) within bacterial-domain reads. B) Host genomic DNA fraction across patients, with each point representing one sequenced swab and point color indicating body region; for patients with more than three samples, boxplots summarize within-patient distributions (center line, median; box, interquartile range [IQR]; whiskers, 1.5 x IQR). C) Focused host-fraction distributions for the two key wound-level signals retained in the adjusted model, comparing acute-like versus other chronicity states and upper-extremity versus other body regions; boxplots are defined as in panel B. D) Fixed-effect estimates from the Gaussian mixed model with patient and culture-date random intercepts, shown as coefficients with 95% confidence intervals on the logit host-fraction scale.

Figure S1. Additional diagnostics for the host-fraction Gaussian mixed-model framework. A) xxx. B) Host-fraction barplot by patient, where host fraction is defined as Homo sapiens reads divided by total Bracken species-level reads for each sample; bar color indicates body region. C) Estimated variance components for patient and culture-date random effects, showing non-trivial structured variance from both sources. D) Boundary-corrected likelihood-ratio tests for random-effect inclusion across nested model contrasts. E) Estimated patient random intercepts from the fitted mixed model, showing broad patient-to-patient spread after fixed-effect adjustment. F) Estimated culture-date random intercepts, demonstrating residual culture-date heterogeneity in host genomic DNA carryover.

### Metagenomic similarity is associated with patient identity and wound chronicity.

To probe the functional relevance of wound microbiome, we sought to evaluate the similarity of the microbial profiles across samples. In descriptive Bray-Curtis analysis, same-patient/same-site/close-date pairs (<6 months apart) were more similar than different-patient pairs (median distance 0.76 vs 0.88, BH-adjusted q = 0.026). Same-patient/close-date pairs (median 0.83, q = 0.0062) and same-patient/same-site pairs (median 0.74, q = 0.0073) were also significantly closer than different-patient pairs, whereas same-patient/different-date-site pairs (median 0.91, q = 0.70) were not (Fig. 2A). We then applied a pairwise mixed-effect model to control for potential batch effect sources during sample collection and sequencing library preparation. In this adjusted model, same-patient pairing (q = 0.031) and same chronicity (q = 9.8e-8) remained associated with higher similarity, while same body region was weaker but still suggestive (q = 0.090) (Fig. 2B, Fig. S2). Technical factors also contributed, with mean host fraction (q = 5.4e-4) and bacterial read depth (q = 0.090), whereas same batch date (q = 0.65) and elapsed-time gap (q = 0.51) were not independently significant (Fig. S2). Model-adjusted margins were directionally consistent, with lower predicted Bray-Curtis distance for shared vs non-shared patient (0.74 vs 0.80), chronicity (0.73 vs 0.80), and body region (0.78 vs 0.80), supporting residual biological structuring after technical adjustment (Fig. 2B).

> Tables:
> - table_03_02_pairwise_distance_summary.tsv (distance average)
> - table_12_02_pairwise_similarity_mixed_effects.tsv
> - table_12_04_pairwise_adjusted_margins.tsv
>
> Figures:
> - fig_03_01_pairwise_distance.svg -> Fig. 2A
> - fig_12_02_pairwise_similarity_mixed.svg -> Fig. S2A
> - fig_12_03_pairwise_adjusted_margins.svg -> Fig. 2B
>

#### Figure captions

Figure 2. Wound microbiome similarity is most strongly associated with shared patient identity and shared chronicity. A) Descriptive Bray-Curtis distance distributions across five pairwise groups: same patient/same site/close date (<6 months), same patient/close date, same patient/same site, same patient/different date-site, and different patient; BH-adjusted q values are shown for four one-sided contrasts against the different-patient reference. Boxplots show median (center line), IQR (box), and 1.5 x IQR whiskers, with overlaid points representing pairwise comparisons. B) Covariate-adjusted marginal Bray-Curtis predictions from the pairwise mixed-effects model, comparing shared versus non-shared states for patient, chronicity, and body region while holding technical covariates and other pairwise indicators fixed.

Figure S2. Adjusted pairwise mixed-effects coefficient summary for metagenomic similarity. A) Fixed-effect estimates (with 95% confidence intervals) from the body-region pairwise mixed model; negative coefficients indicate lower Bray-Curtis distance (greater similarity), and point color indicates BH-adjusted significance.

### Shotgun metagenomic detection agrees with culture in a nonparametric descriptive analysis and remains significant after rank-based technical adjustment

Comparing metagenomic sequencing and bacterial culture results, 10/22 samples with *P. aeruginosa* group abundance over 50% by sequencing had cultures negative for *P. aeruginosa*. Of the 18 samples with *P. aeruginosa* detected by culture-based methods, 17 underwent metagenomic sequencing; most of the sequenced samples (12/17) had the corresponding metagenomic *Pseudomonas* group abundance above 50%.

Taking samples with or without positive culture results and comparing the corresponding microbial abundance in the shotgun metagenomic sequencing data, we found strong descriptive agreements for 7 of 9 organism groups by one-sided Mann-Whitney U test (BH q < 0.1), including the *P. aeruginosa* group (q = 5.0e-5), *Staphylococcus aureus* (q = 5.5e-4), *Klebsiella* spp. (q = 0.0021), *Serratia* (q = 0.0015), GAS (q = 7.3e-5), *Acinetobacter baumannii* (q = 0.031), and *E. faecalis* (q = 0.066), whereas *E. coli* (q = 0.44) and *Proteus mirabilis* (q = 0.15) did not pass the descriptive significance threshold (Fig. 3C). To further verify agreement between culture and metagenomics while accounting for technical and repeated-measures structure, we fit a rank-based mixed model correcting for host fraction, sequencing depth, patient, and batch. In this adjusted force-all-target analysis, culture-positive status remained significant for 8 of 9 groups (BH q < 0.1), with strongest support for GAS (q = 1.2e-7), *Klebsiella* spp. (q = 2.0e-5), and the *P. aeruginosa* group (q = 7.8e-6), while *E. faecalis* was not significant (q = 0.72) (Fig. 3B). Together, these results indicate that shotgun metagenomic detection is concordant with culture for major wound-associated organisms, while low-prevalence targets remain less stable and require larger culture-positive sample counts for robust inference.

> Tables:
> - table_04_01_culture_concordance.tsv
> - table_13_01_culture_threshold_sweep.tsv
> - table_13_02_culture_optimal_thresholds.tsv
> - table_13_03_culture_mixed_concordance.tsv
> - table_13_04_culture_venn_counts.tsv
>
> Figures:
> - fig_13_04_culture_adjusted_concordance.svg -> Fig. 3B
> - fig_13_03_culture_abundance_density.svg -> Fig. 3C
>

#### Figure captions

Figure 3. Shotgun metagenomic abundance is concordant with culture positivity in both descriptive and adjusted analyses. A) xxx. B) Technical/nuisance-adjusted rank-based mixed-model effect sizes for culture-positive versus culture-negative status across all nine organism groups, shown with 95% confidence intervals; colors indicate BH-adjusted significance and marker shape distinguishes regular versus singular fits. C) Organism-level abundance distributions (log10 relative abundance) stratified by culture status, with panel titles reporting BH-adjusted Mann-Whitney q values; boxplots show median (center line), IQR (box), and 1.5 x IQR whiskers, with outliers plotted individually. These descriptive distributions show strongest separation for major wound-associated taxa, including the *P. aeruginosa* group, *Staphylococcus aureus*, *Klebsiella* spp., *Serratia*, and Group A *Streptococcus* (GAS).


## Third party packaged used (including both python and R packages)

### Python

- `pandas`: tabular data loading, cleaning, reshaping, and summary tables.
- `numpy`: numerical arrays, vectorized transformations, and matrix-style operations.
- `scipy`: nonparametric tests, distributions, and likelihood-ratio p-value calculations.
- `statsmodels`: Gaussian mixed models, regression, and multiple-testing correction.
- `matplotlib`: base figure generation and panel layout.
- `seaborn`: statistical plotting layers such as boxplots, violins, and distribution plots.
- `openpyxl`: reading Excel metadata workbooks.
- `nbformat`: notebook generation and structured notebook writing.

### R

- `readr`: TSV/CSV import and export.
- `dplyr`: data wrangling and grouped summaries.
- `tidyr`: reshaping data between wide and long formats.
- `ggplot2`: publication-style figure generation.
- `stringr`: string parsing and text cleaning.
- `vegan`: PERMANOVA and ecological distance-based community analysis.
- `lmerTest`: linear mixed-effects modeling with inferential summaries.
- `broom.mixed`: tidying mixed-model outputs into analysis tables.
- `emmeans`: estimated marginal means and contrast summaries.

## Reproducibility

The following notebooks generate the figures and tables for the 3 results sections above.

### Question 1: Host genomic DNA fraction

- `02_qc_and_host_burden.ipynb` -> `table_02_01_qc_metrics.tsv`, `table_02_02_host_model.tsv`
- `06_mixed_models_and_repeated_measures.ipynb` -> `fig_06_01_host_model_compare.svg`, `table_06_01_patient_structure_diagnostics.tsv`, `table_06_02_host_mixed_effects.tsv`, `table_06_03_host_mixed_status.tsv`
- `11_host_fraction_beta_binomial.ipynb` -> `table_11_01_host_beta_binomial_effects.tsv`, `table_11_02_host_beta_binomial_status.tsv`
- `14_host_fraction_gaussian_story_figures.ipynb` -> `fig_14_01_host_fraction_overview.svg`, `fig_14_02_host_gaussian_mixed_summary.svg`, `fig_14_03_host_gaussian_followup.svg`, `fig_14_04_host_gaussian_random_intercepts.svg`, `table_14_01_host_gaussian_followup.tsv`, `table_14_02_host_gaussian_random_effects.tsv`
- `15_barplots.ipynb` -> `fig_15_01_host_fraction_by_patient.svg`, `fig_15_02_bacterial_genus_by_patient.svg`

### Question 2: Metagenomic similarity by shared covariates

- `03_bracken_community_structure.ipynb` -> `fig_03_01_pairwise_distance.svg`, `table_03_01_pairwise_distances.tsv`, `table_03_02_pairwise_distance_summary.tsv`
- `12_adjusted_community_structure_analysis.ipynb` -> `table_12_01_adjusted_community_permanova.tsv`, `table_12_02_pairwise_similarity_mixed_effects.tsv`, `table_12_03_pairwise_similarity_model_status.tsv`, `table_12_04_pairwise_adjusted_margins.tsv`, `table_12_05_pairwise_similarity_mixed_effects_exact_location_sensitivity.tsv`, `table_12_06_pairwise_similarity_model_status_exact_location_sensitivity.tsv`, `table_12_07_pairwise_adjusted_margins_exact_location_sensitivity.tsv`
- `12_adjusted_community_structure_plots.ipynb` -> `fig_12_01_adjusted_permanova.svg`, `fig_12_02_pairwise_similarity_mixed.svg`, `fig_12_03_pairwise_adjusted_margins.svg`, `fig_12_04_pairwise_similarity_mixed_exact_location_sensitivity.svg`, `fig_12_05_pairwise_adjusted_margins_exact_location_sensitivity.svg`

### Question 3: Culture versus shotgun metagenomics concordance

- `04_culture_concordance.ipynb` -> `fig_04_01_culture_concordance.svg`, `table_04_01_culture_concordance.tsv`
- `13_culture_threshold_and_concordance_analysis.ipynb` -> `table_13_01_culture_threshold_sweep.tsv`, `table_13_02_culture_optimal_thresholds.tsv`, `table_13_03_culture_mixed_concordance.tsv`, `table_13_04_culture_venn_counts.tsv`, `table_13_05_culture_concordance_descriptive.tsv`, `table_13_06_culture_abundance_plot_data.tsv`
- `13_culture_threshold_and_concordance_plots.ipynb` -> `fig_13_01_culture_threshold_sweep.svg`, `fig_13_02_culture_venn_diagrams.svg`, `fig_13_03_culture_abundance_density.svg`, `fig_13_04_culture_adjusted_concordance.svg`

### Shared or Supporting Workflow Components

- `00_get_count_matrix.ipynb` -> `table_00_01_read_count_species_all.csv`, `table_00_02_read_count_species_bac.csv`, `table_00_03_read_count_metaphlan.tsv`, `table_00_04_count_matrix_equivalence.tsv`
- `06_mixed_models_and_repeated_measures.ipynb` -> `fig_06_02_species_mixed_effects.svg`, `table_06_04_species_mixed_effects.tsv`, `table_06_05_species_mixed_status.tsv`, `table_06_06_mixed_vs_cluster_comparison.tsv`
- `07_bracken_vs_metaphlan_sensitivity.ipynb` -> `table_07_01_bracken_metaphlan_sample_depth.tsv`, `table_07_02_bracken_metaphlan_distance_summary.tsv`, `table_07_03_bracken_metaphlan_taxon_correlations.tsv`, `table_07_04_bracken_species_models_shared_samples.tsv`, `table_07_05_metaphlan_species_models.tsv`, `table_07_06_bracken_metaphlan_model_comparison.tsv`
- `08_halla_exploration.ipynb` -> `fig_08_01_halla_top25.svg`, `table_08_01_halla_method_status.tsv`, `table_08_02_halla_top_pairwise_associations.tsv`, `table_08_03_halla_significant_clusters.tsv`
- `09_maaslin2_multivariable.ipynb` -> `fig_09_01_maaslin2_summary.svg`, `table_09_01_maaslin2_model_summary.tsv`, `table_09_02_maaslin2_focus_results.tsv`
- `10_lefse_targeted_contrast.ipynb` -> `fig_10_01_lefse_summary.svg`, `table_10_01_lefse_comparison_summary.tsv`, `table_10_02_lefse_significant_features.tsv`
