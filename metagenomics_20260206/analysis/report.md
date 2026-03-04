# EB shotgun metagenomics analysis

## Scope

- Primary metadata source: `PA_Data_Finalized.xlsx`, sheet `Corrected EB wound spreadsheet`.
- Taxonomic source for bacterial community analysis: Bracken bacterial species counts.
- Host burden source: alignment-based host depletion from `fastp.stats` and `fastp_no_host.stats`, with Bracken human reads used as a cross-check.
- Community-level analyses used a depth-aware QC threshold of at least 10,000 Bracken bacterial reads (61 / 74 samples passed; 13 were retained for descriptive analyses only).

## Key findings

- Host contamination was substantial: median alignment-based host fraction was 94.7%, and median Bracken bacterial fraction among classified species reads was 5.3%.
- Revisit structure is real, not negligible: the 74 swabs came from 18 patients across 41 collection dates spanning 2021-12-02 to 2025-01-31, and 13 of the 18 patients had samples on more than one date.
- Upper-extremity swabs trended toward higher host burden than lower-extremity swabs in the depth-adjusted model (coefficient 1.83 on the logit host scale, q=0.0882), but collection year was an even stronger driver.
- Host burden also shifted strongly over time; the largest collection-year term was `T.2024` with coefficient 4.21 (q=3.37e-07).
- Same-patient, same-visit pairs had median Bray-Curtis distance 0.590, compared with 0.878 for unrelated pairs.
- Same-patient, different-visit pairs were much less stable: median distance was 0.866, essentially similar to unrelated pairs.
- Culture concordance was strongest for S. aureus (AUROC 0.77, q=0.000554; median relative abundance 23.14% in culture-positive swabs).
- Culture concordance was strongest for P. aeruginosa (AUROC 0.87, q=2.43e-05; median relative abundance 69.15% in culture-positive swabs).
- Culture concordance was strongest for Klebsiella spp. (AUROC 0.91, q=0.0021; median relative abundance 28.35% in culture-positive swabs).
- Culture concordance was strongest for GAS (AUROC 0.98, q=7.27e-05; median relative abundance 0.13% in culture-positive swabs).

## Figure captions

1. `figure_01_qc_host_burden.svg`: QC overview. Left, host-depleted read pairs versus Bracken bacterial species reads, with the community-analysis threshold at 10,000 reads. Right, alignment-based host fraction by body region; points are individual swabs colored by patient.
2. `figure_02_pairwise_distance.svg`: Pairwise Bray-Curtis distances between QC-passing swabs. Same-patient, same-visit comparisons are the closest group; same-patient different-visit comparisons are shown separately because revisits occur in this cohort.
3. `figure_03_culture_concordance.svg`: Metagenomic relative abundance of key pathogen groups stratified by whether routine culture called the same organism group. Boxplots summarize distributions; point clouds show individual swabs.
4. `figure_04_species_associations.svg`: Cluster-robust CLR effect sizes for selected taxa versus body region and chronicity covariates after adjusting for sequencing depth.

## Caveats

- Body-site effects should be interpreted jointly with patient and visit structure; repeated measures are common in this cohort.
- Culture agreement is organism-group level, not strain or resistance-level agreement. MRSA and MSSA were collapsed to `S. aureus` because species-level metagenomics does not resolve methicillin resistance.
- Low-bacterial-depth swabs were retained in descriptive host and culture plots but excluded from composition-sensitive community models.

## Species-level model hits highlighted in Figure 4

- Pseudomonas aeruginosa: Chronic-like vs unknown -> effect 2.71 (95% CI 1.29 to 4.13).
- Staphylococcus aureus: Head / neck vs lower extremity -> effect 2.85 (95% CI 1.01 to 4.68).
- Cutibacterium acnes: Head / neck vs lower extremity -> effect 2.15 (95% CI 0.71 to 3.58).
- Serratia marcescens: Acute-like vs unknown -> effect -3.15 (95% CI -5.04 to -1.25).
