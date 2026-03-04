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

### Modeling of host genomic DNA fraction

#### Host fraction definition

Host genomic DNA content was quantified for each sequenced sample as the fraction of trimmed read pairs classified as host after host-removal processing:
$$
h_i \;=\; 1 - \frac{\text{non\_host\_pairs}_i}{\text{trimmed\_pairs}_i}.
$$
Here, $\text{trimmed\_pairs}_i$ was derived from `fastp.stats`, and $\text{non\_host\_pairs}_i$ from `fastp_no_host.stats`. To obtain an approximately unbounded response for Gaussian modeling, host fraction was logit-transformed after truncation away from 0 and 1:
$$
y_i \;=\; \log\!\left(\frac{\tilde h_i}{1-\tilde h_i}\right),
\qquad
\tilde h_i = \min\{\max(h_i,10^{-4}),1-10^{-4}\}.
$$

#### Metadata encoding

All 74 sequenced samples were retained for host-fraction modeling after metadata cleaning and harmonization. `Culture date` was treated as a technical batch variable (`batch_id`). Patient-relative biological time was defined as years since the first sampled date for that patient:
$$
t_i \;=\; \frac{\text{culture\_date}_i - \min(\text{culture\_date for patient }p_i)}{365.25}.
$$
Categorical variables with incomplete clinical annotation were not numerically imputed; instead, uncertainty was retained as explicit levels such as `unknown`. Fixed-effect covariates were encoded with the following reference levels:
- `body_region`: `lower_extremity`
- `chronicity_group`: `unknown`
- `culture_positive_label`: `negative`

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
Here:
- $u_{p(i)} \sim N(0,\sigma^2_{\text{patient}})$ is the patient-specific random intercept
- $b_{k(i)} \sim N(0,\sigma^2_{\text{batch}})$ is the culture-date batch random intercept
- $\varepsilon_i \sim N(0,\sigma^2)$ is the residual error

`Patient` and `batch` were modeled as random effects because they define correlation structure among observations rather than scientific contrasts of interest. `Body region`, `chronicity`, broad culture positivity, and patient-relative elapsed time were modeled as fixed effects because the goal was to estimate interpretable cohort-level associations for those variables.

Two nested fixed-effect specifications were fit:
$$
\texttt{host\_base: } y_i \sim \text{body\_region} + \text{culture\_positive} + t_i + (1|\text{patient}) + (1|\text{batch}),
$$
$$
\texttt{host\_extended: } y_i \sim \text{body\_region} + \text{chronicity} + \text{culture\_positive} + t_i + (1|\text{patient}) + (1|\text{batch}).
$$

#### Model fitting and inference

Models were fit by maximum likelihood (`reml = FALSE`) so that likelihood-ratio tests between nested models were valid. To reduce dependence on a single optimizer, each model was attempted with `lbfgs`, `powell`, and `bfgs`, and the lowest-AIC converged fit was retained. If no converged fit was available, the lowest-AIC available fit was used.

Coefficient-level fixed-effect inference was based on Wald tests from the fitted Gaussian mixed model:
$$
z_j = \frac{\hat\beta_j}{\operatorname{SE}(\hat\beta_j)}.
$$
Two-sided p values were taken from the asymptotic normal approximation implemented in `statsmodels`. Multiple-testing adjustment used the Benjamini-Hochberg false discovery rate (BH-FDR) procedure. Two adjustment schemes were considered:

- `Global BH`: BH correction across all fixed-effect coefficients in the host model,
$$
(q_1,\dots,q_m)=\mathrm{BH}(p_1,\dots,p_m).
$$

- `Factor-specific post hoc BH`: BH correction applied only within the coefficients of a given multi-level factor, for example
$$
(q^{\text{chronicity}}_1,\dots,q^{\text{chronicity}}_k)
=
\mathrm{BH}(p^{\text{chronicity}}_1,\dots,p^{\text{chronicity}}_k).
$$

Significance thresholds were defined as $p < 0.05$ for prespecified single tests and $q < 0.10$ for FDR-adjusted results.

Random-effect terms were assessed by nested likelihood-ratio tests comparing models with and without the relevant variance component. For a nested comparison,
$$
\Lambda = 2\{\ell(\text{full}) - \ell(\text{reduced})\}.
$$
Because the null hypothesis for a variance component is on the boundary of the parameter space,
$$
H_0:\sigma^2 = 0,
$$
p values were computed using the standard one-parameter boundary correction:
$$
p_{\text{boundary}} = \tfrac{1}{2}P(\chi^2_1 \ge \Lambda).
$$

To test whether an entire categorical factor contributed to host fraction, we used likelihood-ratio omnibus tests comparing the full model to reduced models with the factor removed. For `chronicity_group`, the null hypothesis was:
$$
H_0:\gamma_{\text{acute}}=\gamma_{\text{chronic}}=\gamma_{\text{mixed}}=0.
$$
For `body_region`, the null hypothesis was:
$$
H_0:\beta_{\text{head/neck}}=\beta_{\text{upper extremity}}=\beta_{\text{trunk/perineum}}=0.
$$
P values were obtained from a $\chi^2_d$ distribution with $d$ equal to the number of removed coefficients.

To increase power for biologically targeted hypotheses, we also fit prespecified 1-degree-of-freedom contrast models. These replaced the original multi-level factor with a binary indicator corresponding to the contrast of interest.

For acute-like chronicity:
$$
y_i \sim \text{body\_region} + I(\text{acute\_like}) + \text{culture\_positive} + t_i + (1|\text{patient}) + (1|\text{batch}).
$$

For upper-extremity location:
$$
y_i \sim I(\text{upper\_extremity}) + \text{chronicity} + \text{culture\_positive} + t_i + (1|\text{patient}) + (1|\text{batch}).
$$

These planned-contrast p values were BH-adjusted only within the planned-contrast family.

### Similarity analysis of shotgun metagenomic sequencing

#### Outcome definition and distance metric

To assess whether samples sharing biological covariates were more similar in shotgun metagenomic composition, we analyzed Bracken-derived bacterial community profiles from QC-passing samples and quantified between-sample compositional dissimilarity using Bray-Curtis distance. If $p_{is}$ denotes the relative abundance of species $s$ in sample $i$, Bray-Curtis distance between samples $i$ and $j$ was defined as
$$
d_{ij}=\frac{\sum_s |p_{is}-p_{js}|}{\sum_s (p_{is}+p_{js})}.
$$
Distances range from 0 for identical communities to 1 for maximally dissimilar communities. All community-level analyses in this section were restricted to the QC-passing sample set used for composition-sensitive inference.

#### Descriptive pairwise comparisons

As an initial descriptive analysis, all sample pairs were partitioned into three groups: `same patient, same batch date`, `same patient, different batch date`, and `different patient`. Median Bray-Curtis distances for the two within-patient groups were compared against the `different patient` reference group using one-sided Mann-Whitney tests with the alternative hypothesis that within-group distances were smaller. Resulting p values were adjusted with the Benjamini-Hochberg false discovery rate procedure.

#### Multivariable PERMANOVA

To test whether sample-level covariates explained global community variation after mutual adjustment, we fit a multivariable PERMANOVA with Bray-Curtis distance as the response:
$$
\text{Bray-Curtis} \sim \text{batch\_id} + \text{host fraction} + \log_{10}(\text{bacterial reads}) + \text{years since first patient sample} + \text{body region} + \text{chronicity} + \text{culture positivity}.
$$
This was implemented with `adonis2` using `by = "margin"`, so each term was tested marginally while conditioning on the others in the same model. Permutations were stratified by `patient_id` to preserve within-patient structure. For each term, we recorded the partial $R^2$, pseudo-$F$ statistic, p value, and BH-adjusted q value.

#### Pairwise mixed-effects similarity models

Because the biological question is inherently pairwise, we also modeled Bray-Curtis distance directly at the sample-pair level. The pairwise dataset included all unordered pairs of QC-passing samples. For each pair $(i,j)$, we defined binary indicators for whether the two samples shared the same patient, batch date, broad body region, exact location, chronicity group, or culture-positivity status. We also defined continuous pairwise nuisance covariates as the mean host fraction,
$$
\overline{h}_{ij}=\frac{h_i+h_j}{2},
$$
the mean bacterial read depth,
$$
\overline{z}_{ij}=\frac{\log_{10}(\text{bacterial reads}_i)+\log_{10}(\text{bacterial reads}_j)}{2},
$$
and the absolute elapsed-time difference,
$$
\Delta t_{ij}=|t_i-t_j|,
$$
where $t_i$ is years since the first sample from the same patient.

Two linear mixed-effects models were fit. In the first, site similarity was represented at the level of broad anatomical region:
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
In the second, `same body region` was replaced by `same exact location`:
$$
d_{ij}
=
\beta_0
+
\beta_1 I(\text{same patient}_{ij})
+
\beta_2 I(\text{same batch}_{ij})
+
\beta_3 I(\text{same exact location}_{ij})
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
Here, $a_i$ and $a_j$ are crossed random intercepts for the two samples contributing to the pair, and $\varepsilon_{ij}$ is residual error. This crossed random-effects structure accounts for non-independence arising because each sample contributes to many pairwise distances.

#### Fixed-effect inference and interpretation

For the pairwise mixed models, coefficient-level inference was based on Wald tests. For each coefficient $\beta_k$, the test statistic was
$$
t_k = \frac{\hat\beta_k}{\operatorname{SE}(\hat\beta_k)},
$$
with two-sided p values obtained from the fitted mixed model. P values across coefficients within each pairwise model family were adjusted using the Benjamini-Hochberg procedure to generate q values. Significance for FDR-controlled inference was interpreted at $q < 0.10$, while nominal p values were also reported.

The PERMANOVA and pairwise mixed models were interpreted as complementary. PERMANOVA tested whether a covariate explained global community variation after mutual adjustment, whereas the pairwise mixed models addressed the more specific question of whether samples sharing a given biological covariate were more similar after controlling for technical factors. Broad `body_region` and `exact location` were evaluated separately and were not treated as interchangeable site definitions.

### Comparison between shotgun metagenomic sequencing and culture results

#### Organism-group abundance definition

We compared culture results against Bracken-derived bacterial relative abundances aggregated at clinically relevant organism-group level. For each sample $i$, species-level Bracken bacterial counts $x_{is}$ were converted to relative abundances
$$
p_{is} = \frac{x_{is}}{\sum_{s'} x_{is'}},
$$
and organism-group abundance was defined as the sum across the species assigned to that group:
$$
A_i^{(g)} = \sum_{s \in g} p_{is}.
$$
For example, `S. aureus` was represented by `Staphylococcus aureus`, whereas `Klebsiella spp.` was represented by the sum of `Klebsiella pneumoniae` and `Klebsiella oxytoca`.

#### Descriptive nonparametric concordance analysis

The primary descriptive concordance analysis compared organism-group abundance between culture-positive and culture-negative samples using a one-sided Mann-Whitney U test with alternative hypothesis `culture positive > culture negative`. For each organism group, if $A_i$ denotes the relative abundance in sample $i$, the null hypothesis was
$$
H_0: A_i^{(+)} \text{ and } A_i^{(-)} \text{ come from the same distribution},
$$
against the one-sided alternative that abundances were stochastically larger in culture-positive samples. The U statistic was converted into an AUROC-style discrimination summary as
$$
\mathrm{AUROC} = \frac{U}{n_{+}n_{-}},
$$
where $n_{+}$ and $n_{-}$ are the numbers of culture-positive and culture-negative samples, respectively. Benjamini-Hochberg correction was applied across organism groups to obtain q values. This descriptive analysis used all 74 sequenced samples.

#### Threshold sweep and overlap summaries

To characterize practical detection agreement across abundance cutoffs, we evaluated thresholds from `0` to `0.10` relative abundance in increments of `0.001`. At each threshold, sequencing calls were defined as positive if $A_i^{(g)}\ge \tau$. For each organism group and threshold, we computed true positives, false positives, false negatives, and true negatives, then derived sensitivity, specificity, positive predictive value, negative predictive value, F1 score, and Cohen’s kappa. The threshold with the highest F1 score, breaking ties by kappa and then by smaller threshold, was retained as the optimal threshold for each organism group.

#### Adjusted rank-based mixed-effects concordance model

To adjust for technical nuisance structure while remaining nonparametric with respect to abundance scale, we fit a rank-based linear mixed model for each organism group. These models were restricted to samples with at least 5,000 bacterial Bracken reads. For each organism-group abundance $A_i^{(g)}$, we computed its within-dataset rank
$$
r_i = \operatorname{rank}(A_i^{(g)}),
$$
using average ranks for ties, and then transformed ranks to a normal-score scale:
$$
z_i = \Phi^{-1}\left(\frac{r_i - 0.5}{n}\right),
$$
where $\Phi^{-1}$ is the inverse standard normal cumulative distribution function and $n$ is the number of samples in the filtered dataset. The adjusted model was
$$
z_i
=
\beta_0
+
\beta_1 I(\text{culture positive}_i)
+
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
with
$$
u_{\text{patient}(i)} \sim N(0,\sigma^2_{\text{patient}}), \qquad
b_{\text{batch}(i)} \sim N(0,\sigma^2_{\text{batch}}), \qquad
\varepsilon_i \sim N(0,\sigma^2).
$$
Here, `culture positive` was the fixed effect of interest, while `host fraction` and `log10 bacterial reads` were nuisance fixed effects. `Patient` and `batch` were modeled as random intercepts to account for repeated sampling within patient and technical batch structure associated with culture date. For each organism group, models were skipped if there were fewer than 4 culture-positive samples, fewer than 4 culture-negative samples, or no variation in the rank-normalized abundance response. Fixed-effect p values for culture status were taken from Wald tests and adjusted across organism groups by Benjamini-Hochberg FDR.

## Results

### Host genomic DNA content in wound skin swabs is associated with structured batch effects and acute-like wound status

Skin swabs are know to. To separate biology from nuisance structure, we modeled host genomic DNA content as logit-transformed host fraction in a Gaussian mixed model with `patient` and `culture-date batch` treated as structured nuisance effects and `body_region`, `chronicity_group`, `culture_positive`, and `years_since_first_sample` treated as fixed effects. Both nuisance terms contributed materially to the model: relative to a fixed-effects-only model, adding `patient` improved fit (`LRT = 8.39`, `p = 0.00189`) and adding `batch` improved fit (`LRT = 5.48`, `p = 0.00964`); each also remained informative when added on top of the other (`batch added to patient: LRT = 3.72`, `p = 0.0268`; `patient added to batch: LRT = 6.64`, `p = 0.00499`). After controlling for these structured effects, `acute-like` wounds remained the clearest biological signal. In the full mixed model, the `acute_like` coefficient was positive (`beta = 2.32`, Wald `p = 0.0284`), and in a prespecified contrast comparing `acute_like` against all other chronicity classes, the association remained significant (`beta = 1.85`, `p = 0.00998`, `q = 0.0200`, Benjamini-Hochberg across planned contrasts). Within the full model, factor-specific BH correction across chronicity terms also supported this effect (`q = 0.0853`). `Upper_extremity` samples showed a positive but weaker association with host fraction and were therefore treated as suggestive rather than definitive (`beta = 1.34`, Wald `p = 0.0646` in the full model). By contrast, other covariates, including broad culture positivity, elapsed time since first patient sample, and the remaining chronicity and location categories, did not provide convincing independent explanatory power *once patient-specific and experiment-batch effects were taken into account*.

### Patient identity and wound chronicity remain associated with metagenomic similarity after technical adjustment

Raw Bray-Curtis distances showed that samples collected from the same patient on the same batch date were substantially more similar than samples from different patients (median distance `0.5897` vs `0.8777`, BH-adjusted `q = 0.000158`), whereas this similarity weakened across different batch dates within the same patient (median distance `0.8660`, `q = 0.112`) [table_03_02_pairwise_distance_summary.tsv](tables/table_03_02_pairwise_distance_summary.tsv). In the multivariable PERMANOVA, no term passed FDR after mutual adjustment, although batch date was the largest near-threshold contributor (`R^2 = 0.528`, `p = 0.0145`, `q = 0.1015`); host fraction (`R^2 = 0.0226`, `p = 0.0680`, `q = 0.119`), bacterial read depth (`R^2 = 0.0212`, `p = 0.0620`, `q = 0.119`), and years since first patient sample (`R^2 = 0.0206`, `p = 0.0580`, `q = 0.119`) also contributed more than the biological covariates, whereas body region (`p = 0.910`, `q = 0.910`), chronicity (`p = 0.207`, `q = 0.2898`), and culture positivity (`p = 0.3305`, `q = 0.3856`) were not globally significant [table_12_01_adjusted_community_permanova.tsv](tables/table_12_01_adjusted_community_permanova.tsv). However, the higher-resolution pairwise mixed-effects models showed that, after correcting for mean host fraction, mean bacterial read depth, batch matching, elapsed-time gap, and culture-positivity matching, similarity remained significantly associated with shared patient identity and shared chronicity. In the body-region model, `same patient` was associated with lower Bray-Curtis distance (`estimate = -0.0607`, `p = 0.0115`, `q = 0.0305`) and `same chronicity` was even stronger (`estimate = -0.0691`, `p = 1.25e-08`, `q = 1.00e-07`); the same pattern held in the exact-location model (`same patient`: `estimate = -0.0648`, `p = 0.00686`, `q = 0.0183`; `same chronicity`: `estimate = -0.0697`, `p = 9.48e-09`, `q = 7.58e-08`) [table_12_02_pairwise_similarity_mixed_effects.tsv](tables/table_12_02_pairwise_similarity_mixed_effects.tsv). Technical factors remained important in the same models, particularly mean host fraction (body-region model: `estimate = -0.441`, `p = 3.36e-05`, `q = 1.35e-04`; exact-location model: `estimate = -0.443`, `p = 2.89e-05`, `q = 1.15e-04`) and bacterial read depth (body-region model: `estimate = -0.0740`, `p = 0.0296`, `q = 0.0592`; exact-location model: `estimate = -0.0748`, `p = 0.0273`, `q = 0.0547`), whereas same batch date itself was not independently associated with similarity (`p = 0.648`, `q = 0.668` and `p = 0.642`, `q = 0.706`, respectively). Broad body region showed only a modest, suggestive association with greater similarity (`estimate = -0.0231`, `p = 0.0555`, `q = 0.0889`), while same exact location was not supported after adjustment (`estimate = +0.0145`, `p = 0.706`, `q = 0.706`). Together, these results indicate that patient identity and wound chronicity explain residual metagenomic similarity more consistently than exact anatomical site once technical variation is taken into account.

### Shotgun metagenomic detection agrees with culture in a nonparametric descriptive analysis and remains significant after rank-based technical adjustment

We evaluated agreement between wound culture and shotgun metagenomic sequencing at the organism-group level using two complementary layers of analysis: a descriptive nonparametric comparison of organism abundance between culture-positive and culture-negative samples, and a nuisance-adjusted rank-based mixed-effects model that accounted for patient, batch, host burden, and bacterial depth. In the descriptive analysis across all 74 sequenced samples, culture-positive samples consistently showed higher shotgun abundance than culture-negative samples for several clinically important organisms by one-sided Mann-Whitney U testing. The strongest organism-group agreements were observed for `P. aeruginosa` (`U = 839`, AUROC `0.866`, `q = 2.4e-05`), `S. aureus` (`U = 854`, AUROC `0.767`, `q = 5.5e-04`), `Klebsiella spp.` (`U = 314`, AUROC `0.910`, `q = 0.00210`), `Serratia` (`U = 275`, AUROC `0.982`, `q = 0.00146`), and `GAS` (`U = 274`, AUROC `0.979`, `q = 7.3e-05`) [table_04_01_culture_concordance.tsv](tables/table_04_01_culture_concordance.tsv). Threshold sweeps from `0%` to `10%` relative abundance showed that the abundance threshold yielding the best F1 score differed by organism, with optimal cutoffs of `0.4%` for `S. aureus`, `9.8%` for `P. aeruginosa`, `9.7%` for `Klebsiella spp.`, and `7.3%` for `Serratia` [table_13_02_culture_optimal_thresholds.tsv](tables/table_13_02_culture_optimal_thresholds.tsv). To test whether culture-versus-metagenomics agreement persisted after correcting for technical nuisance structure, we then fit rank-based mixed models on the subset of 68 samples with at least 5,000 bacterial Bracken reads. In these models, the response was a normal-score transform of the rank of organism-group relative abundance, and fixed effects included culture status, host fraction, and bacterial read depth, with random intercepts for patient and batch. Under this adjusted analysis, culture-positive status remained strongly associated with higher metagenomic abundance for `GAS` (estimate `2.113`, `q = 6.93e-08`), `P. aeruginosa` (estimate `1.444`, `q = 9.59e-07`), `Serratia` (estimate `1.300`, `q = 3.02e-04`), `S. aureus` (estimate `1.112`, `q = 1.10e-05`), and `Klebsiella spp.` (estimate `1.966`, `q = 1.10e-05`) [table_13_03_culture_mixed_concordance.tsv](tables/table_13_03_culture_mixed_concordance.tsv). Rare targets such as `Proteus`, `E. coli`, `A. baumannii`, and `E. faecalis` remained too sparse for stable adjusted inference and were skipped. Together, these results indicate that shotgun metagenomic detection is concordant with culture for several major wound-associated organisms and that this agreement persists even after nonparametric abundance ranking and correction for patient-specific, batch, host, and depth effects.

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
- `emmeans`: estimated marginal means and planned contrasts.

## Reproducibility

The following notebooks generate the figures and tables for the 3 results sections above.

### Question 1: Host genomic DNA fraction

- `02_qc_and_host_burden.ipynb` -> `fig_02_01_qc_host_burden.svg`, `table_02_01_qc_metrics.tsv`, `table_02_02_host_model.tsv`
- `06_mixed_models_and_repeated_measures.ipynb` -> `fig_06_01_host_model_compare.svg`, `table_06_01_patient_structure_diagnostics.tsv`, `table_06_02_host_mixed_effects.tsv`, `table_06_03_host_mixed_status.tsv`
- `11_host_fraction_beta_binomial.ipynb` -> `fig_11_01_host_beta_binomial.svg`, `table_11_01_host_beta_binomial_effects.tsv`, `table_11_02_host_beta_binomial_status.tsv`
- `14_host_fraction_gaussian_story_figures.ipynb` -> `fig_14_01_host_fraction_overview.svg`, `fig_14_02_host_gaussian_mixed_summary.svg`, `fig_14_03_host_gaussian_followup.svg`, `fig_14_04_host_gaussian_random_intercepts.svg`, `table_14_01_host_gaussian_followup.tsv`, `table_14_02_host_gaussian_random_effects.tsv`

### Question 2: Metagenomic similarity by shared covariates

- `03_bracken_community_structure.ipynb` -> `fig_03_01_pairwise_distance.svg`, `table_03_01_pairwise_distances.tsv`, `table_03_02_pairwise_distance_summary.tsv`
- `12_adjusted_community_structure.ipynb` -> `fig_12_01_adjusted_permanova.svg`, `fig_12_02_pairwise_similarity_mixed.svg`, `fig_12_03_pairwise_adjusted_margins.svg`, `table_12_01_adjusted_community_permanova.tsv`, `table_12_02_pairwise_similarity_mixed_effects.tsv`, `table_12_03_pairwise_similarity_model_status.tsv`, `table_12_04_pairwise_adjusted_margins.tsv`

### Question 3: Culture versus shotgun metagenomics concordance

- `04_culture_concordance.ipynb` -> `fig_04_01_culture_concordance.svg`, `table_04_01_culture_concordance.tsv`
- `13_culture_threshold_and_concordance.ipynb` -> `fig_13_01_culture_threshold_sweep.svg`, `fig_13_02_culture_venn_diagrams.svg`, `fig_13_03_culture_abundance_density.svg`, `fig_13_04_culture_adjusted_concordance.svg`, `table_13_01_culture_threshold_sweep.tsv`, `table_13_02_culture_optimal_thresholds.tsv`, `table_13_03_culture_mixed_concordance.tsv`, `table_13_04_culture_venn_counts.tsv`

### Shared or Supporting Workflow Components

- `06_mixed_models_and_repeated_measures.ipynb` -> `fig_06_02_species_mixed_effects.svg`, `table_06_04_species_mixed_effects.tsv`, `table_06_05_species_mixed_status.tsv`, `table_06_06_mixed_vs_cluster_comparison.tsv`
- `07_bracken_vs_metaphlan_sensitivity.ipynb` -> `table_07_01_bracken_metaphlan_sample_depth.tsv`, `table_07_02_bracken_metaphlan_distance_summary.tsv`, `table_07_03_bracken_metaphlan_taxon_correlations.tsv`, `table_07_04_bracken_species_models_shared_samples.tsv`, `table_07_05_metaphlan_species_models.tsv`, `table_07_06_bracken_metaphlan_model_comparison.tsv`
- `08_halla_exploration.ipynb` -> `fig_08_01_halla_top25.svg`, `table_08_01_halla_method_status.tsv`, `table_08_02_halla_top_pairwise_associations.tsv`, `table_08_03_halla_significant_clusters.tsv`
- `09_maaslin2_multivariable.ipynb` -> `fig_09_01_maaslin2_summary.svg`, `table_09_01_maaslin2_model_summary.tsv`, `table_09_02_maaslin2_focus_results.tsv`
- `10_lefse_targeted_contrast.ipynb` -> `fig_10_01_lefse_summary.svg`, `table_10_01_lefse_comparison_summary.tsv`, `table_10_02_lefse_significant_features.tsv`
