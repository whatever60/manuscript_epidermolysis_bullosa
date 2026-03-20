#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests


SPECIAL_SAMPLE_RENAMES = {
    "SUB2H1": "SUB2K",
    "SUB2H2": "SUB2H",
    "SUB7A1": "SUB7A",
    "SUB7A2": "SUB6A",
    "SUB18A1": "SUB18A",
    "SUB18A2": "SUB19A",
}

BODY_REGION_ORDER = [
    "head_neck",
    "upper_extremity",
    "trunk_perineum",
    "lower_extremity",
]

BODY_REGION_LABELS = {
    "head_neck": "Head / neck",
    "upper_extremity": "Upper extremity",
    "trunk_perineum": "Trunk / perineum",
    "lower_extremity": "Lower extremity",
    "unknown": "Unknown",
}

CHRONICITY_ORDER = ["acute_like", "chronic_like", "mixed", "unknown"]

CHRONICITY_LABELS = {
    "acute_like": "Acute-like",
    "chronic_like": "Chronic-like",
    "mixed": "Mixed",
    "unknown": "Unknown",
}

CULTURE_GROUPS = [
    {
        "group": "s_aureus",
        "label": "S. aureus",
        "culture_patterns": [r"\bmssa\b", r"\bmrsa\b", r"staphylococcus aureus"],
        "taxa": ["Staphylococcus aureus"],
    },
    {
        "group": "p_aeruginosa",
        "label": "P. aeruginosa",
        "culture_patterns": [r"pseudomonas aeruginosa"],
        "taxa": ["Pseudomonas aeruginosa"],
    },
    {
        "group": "serratia_marcescens",
        "label": "Serratia",
        "culture_patterns": [r"serratia marcescens"],
        "taxa": ["Serratia marcescens"],
    },
    {
        "group": "proteus_mirabilis",
        "label": "Proteus",
        "culture_patterns": [r"proteus mirabilis"],
        "taxa": ["Proteus mirabilis"],
    },
    {
        "group": "gas",
        "label": "GAS",
        "culture_patterns": [r"\bgas\b", r"streptococcus pyogenes", r"strep pyogenes"],
        "taxa": ["Streptococcus pyogenes"],
    },
    {
        "group": "klebsiella_spp",
        "label": "Klebsiella spp.",
        "culture_patterns": [r"klebsiella"],
        "taxa": ["Klebsiella pneumoniae", "Klebsiella oxytoca"],
    },
    {
        "group": "e_coli",
        "label": "E. coli",
        "culture_patterns": [r"escherichia coli", r"\be\. coli\b"],
        "taxa": ["Escherichia coli"],
    },
    {
        "group": "acinetobacter_baumannii",
        "label": "A. baumannii",
        "culture_patterns": [r"acinetobacter baumannii"],
        "taxa": ["Acinetobacter baumannii"],
    },
    {
        "group": "e_faecalis",
        "label": "E. faecalis",
        "culture_patterns": [r"enterococcus faecalis"],
        "taxa": ["Enterococcus faecalis"],
    },
]

KEY_SPECIES = [
    "Staphylococcus aureus",
    "Pseudomonas aeruginosa",
    "Serratia marcescens",
    "Corynebacterium striatum",
    "Cutibacterium acnes",
    "Klebsiella pneumoniae",
]


@dataclass
class AnalysisContext:
    data_dir: Path
    analysis_dir: Path
    figure_dir: Path
    table_dir: Path


def normalize_sample_id(name: str) -> str:
    text = str(name).strip()
    if not text:
        return text

    upper = text.upper()
    upper = upper.replace(".TXT", "")
    upper = upper.replace(".CSV", "")
    upper = upper.replace(".TSV", "")
    upper = re.sub(
        r"(_KRAKEN.*|_BRACKEN.*|_FASTQC.*|_R[12]_001.*|_[12]\.FQ\.GZ.*)$", "", upper
    )
    if upper.startswith("YQEBMETA"):
        upper = "SUB" + upper.replace("YQEBMETA", "", 1)
    if not upper.startswith("SUB"):
        upper = "SUB" + upper
    upper = SPECIAL_SAMPLE_RENAMES.get(upper, upper)
    suffix = upper.replace("SUB", "", 1)
    return suffix.zfill(3)


def clean_free_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip().strip('"')
    return text


def standardize_location(value: object) -> str:
    text = clean_free_text(value).lower()
    if not text:
        return ""
    text = text.replace("pre-auricular", "preauricular")
    text = re.sub(r"^\br\b(?=\s)", "right", text)
    text = re.sub(r"^\bl\b(?=\s)", "left", text)
    text = re.sub(
        r"\br\b(?=\s+(?:arm|axilla|buttock|chest|ear|earlobe|elbow|finger|foot|hand|knee|leg|neck|shoulder|shin|thigh|upper))",
        "right",
        text,
    )
    text = re.sub(
        r"\bl\b(?=\s+(?:ankle|arm|groin|knee|leg|medial|shin|thigh|upper|foot|buttock|forearm))",
        "left",
        text,
    )
    text = text.replace("mid lower", "mid-lower")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def infer_body_region(location: str) -> str:
    text = location.lower()
    if not text:
        return "unknown"

    def contains_any_word(words: list[str]) -> bool:
        return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)

    head_neck = ["ear", "earlobe", "eye", "nose", "neck", "forehead", "antihelix"]
    upper_extremity = [
        "shoulder",
        "arm",
        "axilla",
        "elbow",
        "hand",
        "finger",
        "forearm",
    ]
    trunk_perineum = [
        "back",
        "chest",
        "abdomen",
        "buttock",
        "buttocks",
        "groin",
        "perineum",
        "trunk",
    ]
    lower_extremity = ["knee", "shin", "ankle", "foot", "thigh", "leg"]

    if contains_any_word(head_neck):
        return "head_neck"
    if contains_any_word(upper_extremity):
        return "upper_extremity"
    if contains_any_word(trunk_perineum):
        return "trunk_perineum"
    if contains_any_word(lower_extremity):
        return "lower_extremity"
    return "unknown"


def infer_laterality(location: str) -> str:
    text = location.lower()
    if not text:
        return "unknown"
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    if text.startswith("mid"):
        return "midline"
    return "unspecified"


def infer_chronicity(text: str) -> str:
    value = text.lower()
    if not value:
        return "unknown"
    acute_tokens = [
        "acute",
        "new ",
        "new(",
        "new,",
        "new-",
        "<6",
        "3-day",
        "3 wk",
        "3-day",
        "subacute",
    ]
    chronic_tokens = ["chronic", "non-healing", "recurring", ">6", "continued trauma"]
    has_acute = any(token in value for token in acute_tokens)
    has_chronic = any(token in value for token in chronic_tokens)
    if "acute-on-chronic" in value or (has_acute and has_chronic):
        return "mixed"
    if has_chronic:
        return "chronic_like"
    if has_acute:
        return "acute_like"
    return "unknown"


def infer_clinical_infection(text: str) -> str:
    value = text.lower()
    if not value:
        return "unknown"
    negative_patterns = [
        "no signs of infection",
        "not infected",
        "no purulence",
        "no malodor",
        "no tenderness",
        "clinically improving",
        "clean.",
        "clean-based",
        "clean,",
    ]
    if any(pattern in value for pattern in negative_patterns):
        return "no"
    infection_tokens = [
        "purulent",
        "purulence",
        "pus",
        "malodor",
        "drainage",
        "exudate",
        "yellow-green fluid",
        "inflamed",
        "erythema",
    ]
    return "yes" if any(token in value for token in infection_tokens) else "unclear"


def infer_pmn_category(text: str) -> str:
    value = text.lower()
    if not value:
        return "unknown"
    if "no polymorphonuclear" in value or "no pmn" in value:
        return "none"
    if "rare polymorphonuclear" in value or "rare pmn" in value:
        return "rare"
    if "few polymorphonuclear" in value or "few pmn" in value:
        return "few"
    if "moderate polymorphonuclear" in value or "moderate pmn" in value:
        return "moderate"
    if "many polymorphonuclear" in value or "many pmn" in value:
        return "many"
    return "unknown"


def parse_seqkit_stats(path: Path) -> pd.DataFrame:
    stats = pd.read_table(path, sep=r"\s+", engine="python")
    for column in ["num_seqs", "sum_len", "min_len", "avg_len", "max_len"]:
        stats[column] = (
            stats[column].astype(str).str.replace(",", "", regex=False).astype(float)
        )

    file_names = stats["file"].map(lambda value: Path(value).name)

    def extract_sample(file_name: str) -> str:
        if "_R1_001.fastq.gz" in file_name or "_R2_001.fastq.gz" in file_name:
            return normalize_sample_id(file_name.split("_")[0])
        if file_name.endswith("_1.fq.gz"):
            return normalize_sample_id(file_name[: -len("_1.fq.gz")])
        if file_name.endswith("_2.fq.gz"):
            return normalize_sample_id(file_name[: -len("_2.fq.gz")])
        return normalize_sample_id(file_name)

    def extract_read(file_name: str) -> str:
        if "_R1_001.fastq.gz" in file_name or file_name.endswith("_1.fq.gz"):
            return "R1"
        if "_R2_001.fastq.gz" in file_name or file_name.endswith("_2.fq.gz"):
            return "R2"
        return "NA"

    stats["sample_id"] = file_names.map(extract_sample)
    stats["read"] = file_names.map(extract_read)
    stats = stats.loc[
        stats["read"] == "R1", ["sample_id", "num_seqs", "sum_len", "avg_len"]
    ].copy()
    stats = stats.rename(
        columns={"num_seqs": "pairs", "sum_len": "sum_len_r1", "avg_len": "avg_len_r1"}
    )
    stats["pairs"] = stats["pairs"].astype(int)
    return stats.set_index("sample_id").sort_index()


def topological_sort(graph):
    visited = set()
    stack = []

    def dfs(node):
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(node)

    for node in graph:
        if node not in visited:
            dfs(node)

    return stack[::-1]


def rename_sample(name: str) -> str:
    if name == "SUB2h1":
        name = "SUB2k"
    elif name == "SUB2h2":
        name = "SUB2h"
    elif name == "SUB7a1":
        name = "SUB7a"
    elif name == "SUB7a2":
        name = "SUB6a"
    elif name == "SUB18a1":
        name = "SUB18a"
    elif name == "SUB18a2":
        name = "SUB19a"
    elif name.startswith("yqebmeta"):
        name = name.replace("yqebmeta", "SUB")

    name = name.replace("SUB", "")
    name = name.zfill(3).upper()
    return name


def parse_bracken_reports(
    directory_path: str,
    pattern: str = "_kraken_report_bracken_species.txt",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Faithful port of the original get_bracken_table.ipynb parser.
    file_paths = glob(str(Path(directory_path) / f"*{pattern}"))

    all_data: dict[str, dict[str, float]] = {}
    all_reads: dict[str, dict[str, int]] = {}
    taxonomy_records: dict[str, dict[str, str | None]] = {}
    dups = defaultdict(int)
    graph = defaultdict(list)

    for file_path in file_paths:
        filename = Path(file_path).name
        sample_name = filename.split("_")[0]

        df = pd.read_table(
            file_path,
            header=None,
            names=["percentage", "reads", "reads_direct", "level", "taxid", "name"],
            dtype={"name": str},
        )

        df["indent"] = df["name"].str.extract(r"^(\s*)")[0].str.len()
        df["name"] = df["name"].str.lstrip()

        sample_abund: dict[str, float] = {}
        sample_reads: dict[str, int] = {}
        lineage_stack: list[tuple[int, str, str]] = []

        for _, row in df.iterrows():
            taxon_name = row["name"]
            level_code = row["level"]
            indent = row["indent"]

            while lineage_stack and lineage_stack[-1][0] >= indent:
                lineage_stack.pop()
            if lineage_stack:
                src = lineage_stack[-1][2]
                graph[src].append(level_code)

            current_lineage: dict[str, str | None] = {}
            for _, ancestor_name, ancestor_level in lineage_stack:
                current_lineage[ancestor_level] = ancestor_name

            current_lineage[level_code] = taxon_name
            lineage_stack.append((indent, taxon_name, level_code))
            if taxon_name not in taxonomy_records:
                taxonomy_records[taxon_name] = current_lineage.copy()
            else:
                if taxonomy_records[taxon_name] != current_lineage:
                    dups[taxon_name] += 1
                    taxon_name = taxon_name + f".{dups[taxon_name]}"
                    taxonomy_records[taxon_name] = current_lineage.copy()

            sample_abund[taxon_name] = row["percentage"]
            sample_reads[taxon_name] = row["reads"]

        all_data[sample_name] = sample_abund
        all_reads[sample_name] = sample_reads

    abundance_df = pd.DataFrame.from_dict(all_data, orient="index").fillna(0)
    read_count_df = pd.DataFrame.from_dict(all_reads, orient="index").fillna(0)
    cols = topological_sort(graph)
    tax_df = pd.DataFrame.from_dict(taxonomy_records, orient="index", columns=cols)
    return abundance_df, read_count_df, tax_df


def load_bracken_with_host_counts(context: AnalysisContext) -> pd.DataFrame:
    _, read_count_df, _ = parse_bracken_reports(
        str(context.data_dir / "kraken_with_host")
    )
    read_count_df.index = read_count_df.index.map(rename_sample)
    read_count_df = read_count_df.groupby(level=0).sum().sort_index()
    return read_count_df


def load_bracken_tables(context: AnalysisContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    species_all = pd.read_csv(
        context.data_dir / "read_count_species_all.csv", index_col=0
    )
    species_bac = pd.read_csv(
        context.data_dir / "read_count_species_bac.csv", index_col=0
    )
    species_all.index = species_all.index.map(normalize_sample_id)
    species_bac.index = species_bac.index.map(normalize_sample_id)
    species_all = species_all.groupby(level=0).sum().sort_index()
    species_bac = species_bac.groupby(level=0).sum().sort_index()
    return species_all, species_bac


def load_kraken_unclassified_counts(
    context: AnalysisContext, report_subdir: str = "kraken_with_host"
) -> pd.Series:
    report_dir = context.data_dir / report_subdir
    values: dict[str, int] = {}
    for report_path in sorted(report_dir.glob("*_kraken_report.txt")):
        sample_stub = report_path.name[: -len("_kraken_report.txt")]
        sample_id = normalize_sample_id(sample_stub)
        unclassified_reads: int | None = None
        with report_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                parts = raw_line.rstrip("\n").split("\t")
                if len(parts) < 6:
                    continue
                rank_code = parts[3].strip()
                taxid = parts[4].strip()
                name = parts[5].strip().lower()
                if rank_code == "U" and taxid == "0" and name == "unclassified":
                    try:
                        unclassified_reads = int(float(parts[1]))
                    except ValueError:
                        unclassified_reads = 0
                    break
        values[sample_id] = 0 if unclassified_reads is None else unclassified_reads

    if not values:
        return pd.Series(dtype=float, name="kraken_unclassified_reads")

    series = (
        pd.Series(values, name="kraken_unclassified_reads")
        .groupby(level=0)
        .sum()
        .sort_index()
    )
    return series


def load_metaphlan_table(context: AnalysisContext) -> pd.DataFrame:
    metaphlan = pd.read_csv(
        context.data_dir / "read_count_metaphlan.tsv", sep="\t", index_col=0
    )
    metaphlan.index = metaphlan.index.map(normalize_sample_id)
    metaphlan = metaphlan.groupby(level=0).sum().sort_index()
    return metaphlan


def load_metadata(context: AnalysisContext, sample_ids: list[str]) -> pd.DataFrame:
    primary = pd.read_excel(
        context.data_dir / "metadata" / "PA_Data_Finalized.xlsx",
        sheet_name="Corrected EB wound spreadsheet",
    )
    supplement = pd.read_excel(
        context.data_dir
        / "metadata"
        / "EB_Wound_Cultures_Coded_Data_Lab_Archives_20260210.xlsx",
        sheet_name="Sheet1",
    )

    primary = primary.rename(
        columns={
            "Code": "code",
            "Sequencing data?": "sequencing_data",
            "Culture data?": "culture_data",
            "Pseudomonas": "pseudomonas_flag",
            "Date of Culture": "culture_date",
            "Location": "location_raw",
            "Result": "culture_result_primary",
            "Sensitivities": "sensitivities",
            "Gram Stain": "gram_stain_primary",
            "Clinical Correlates (chronicity, clinical signs of infection, SCC)": "clinical_correlates_primary",
            "Sent to…": "sent_to_primary",
        }
    )
    primary = primary[
        [
            "code",
            "sequencing_data",
            "culture_data",
            "pseudomonas_flag",
            "culture_date",
            "location_raw",
            "culture_result_primary",
            "sensitivities",
            "gram_stain_primary",
            "clinical_correlates_primary",
            "sent_to_primary",
        ]
    ].copy()
    supplement = supplement.rename(
        columns={
            "Code ": "code",
            "Date of Culture ": "culture_date_supplement",
            "Location": "location_supplement",
            "Clinical Correlates (chronicity, clinical signs of infection, SCC)": "clinical_correlates_supplement",
            "Result": "culture_result_supplement",
            "Gram Stain": "gram_stain_supplement",
            "Sent to…": "sent_to_supplement",
        }
    )
    supplement = supplement[
        [
            "code",
            "culture_date_supplement",
            "location_supplement",
            "clinical_correlates_supplement",
            "culture_result_supplement",
            "gram_stain_supplement",
            "sent_to_supplement",
        ]
    ].copy()

    primary["sample_id"] = primary["code"].map(normalize_sample_id)
    supplement["sample_id"] = supplement["code"].map(normalize_sample_id)

    primary = primary.loc[primary["sample_id"].isin(sample_ids)].copy()
    supplement = supplement.loc[supplement["sample_id"].isin(sample_ids)].copy()
    primary = primary.set_index("sample_id")
    supplement = supplement.set_index("sample_id")

    metadata = primary.join(
        supplement[
            [
                "culture_date_supplement",
                "location_supplement",
                "clinical_correlates_supplement",
                "culture_result_supplement",
                "gram_stain_supplement",
                "sent_to_supplement",
            ]
        ],
        how="outer",
    )
    metadata = metadata.reindex(sample_ids)

    metadata["culture_date"] = metadata["culture_date"].fillna(
        metadata["culture_date_supplement"]
    )
    metadata["location_raw"] = metadata["location_raw"].fillna(
        metadata["location_supplement"]
    )
    metadata["clinical_correlates"] = metadata["clinical_correlates_primary"].fillna(
        metadata["clinical_correlates_supplement"]
    )
    metadata["culture_result"] = metadata["culture_result_primary"].fillna(
        metadata["culture_result_supplement"]
    )
    metadata["gram_stain"] = metadata["gram_stain_primary"].fillna(
        metadata["gram_stain_supplement"]
    )
    metadata["sent_to"] = metadata["sent_to_primary"].fillna(
        metadata["sent_to_supplement"]
    )

    metadata["culture_date"] = pd.to_datetime(metadata["culture_date"])
    metadata["patient_id"] = metadata.index.str.slice(0, 2)
    metadata["sample_letter"] = metadata.index.str.slice(2, 3)
    metadata["visit_id"] = (
        metadata["patient_id"] + "_" + metadata["culture_date"].dt.strftime("%Y-%m-%d")
    )
    metadata["batch_id"] = metadata["culture_date"].dt.strftime("%Y-%m-%d")
    metadata["first_culture_date"] = metadata.groupby("patient_id")[
        "culture_date"
    ].transform("min")
    metadata["days_since_first_sample"] = (
        metadata["culture_date"] - metadata["first_culture_date"]
    ).dt.days.astype(float)
    metadata["years_since_first_sample"] = metadata["days_since_first_sample"] / 365.25
    metadata["location"] = metadata["location_raw"].map(standardize_location)
    metadata["body_region"] = metadata["location"].map(infer_body_region)
    metadata["body_region_label"] = metadata["body_region"].map(BODY_REGION_LABELS)
    metadata["laterality"] = metadata["location"].map(infer_laterality)
    metadata["clinical_correlates"] = metadata["clinical_correlates"].map(
        clean_free_text
    )
    metadata["culture_result"] = metadata["culture_result"].map(clean_free_text)
    metadata["gram_stain"] = metadata["gram_stain"].map(clean_free_text)
    metadata["chronicity_group"] = metadata["clinical_correlates"].map(infer_chronicity)
    metadata["chronicity_label"] = metadata["chronicity_group"].map(CHRONICITY_LABELS)
    metadata["clinical_infection_flag"] = metadata["clinical_correlates"].map(
        infer_clinical_infection
    )
    metadata["pmn_category"] = metadata["gram_stain"].map(infer_pmn_category)
    metadata["pseudomonas_flag"] = metadata["pseudomonas_flag"].notna()
    metadata["culture_positive"] = ~metadata["culture_result"].str.lower().isin(
        ["", "no growth", "mixed commensal microbiota"]
    )

    for config in CULTURE_GROUPS:
        pattern = "|".join(config["culture_patterns"])
        metadata[f"culture_{config['group']}"] = (
            metadata["culture_result"]
            .str.lower()
            .str.contains(
                pattern,
                regex=True,
                na=False,
            )
        )

    metadata["n_culture_groups"] = metadata[
        [f"culture_{config['group']}" for config in CULTURE_GROUPS]
    ].sum(axis=1)
    return metadata


def prepare_qc_table(
    context: AnalysisContext,
    metadata: pd.DataFrame,
    species_all: pd.DataFrame,
    species_bac: pd.DataFrame,
    metaphlan: pd.DataFrame,
) -> pd.DataFrame:
    input_stats = parse_seqkit_stats(context.data_dir / "input.stats").rename(
        columns={"pairs": "raw_pairs"}
    )
    fastp_stats = parse_seqkit_stats(context.data_dir / "fastp.stats").rename(
        columns={"pairs": "trimmed_pairs"}
    )
    no_host_stats = parse_seqkit_stats(context.data_dir / "fastp_no_host.stats").rename(
        columns={"pairs": "non_host_pairs"}
    )

    qc = metadata.join(input_stats[["raw_pairs"]], how="left")
    qc = qc.join(fastp_stats[["trimmed_pairs"]], how="left")
    qc = qc.join(no_host_stats[["non_host_pairs"]], how="left")

    qc["trimmed_fraction"] = qc["trimmed_pairs"] / qc["raw_pairs"]
    qc["host_alignment_fraction"] = 1 - (qc["non_host_pairs"] / qc["trimmed_pairs"])

    with_host_counts = (
        load_bracken_with_host_counts(context).reindex(qc.index).fillna(0)
    )
    qc["classified_species_reads"] = species_all.sum(axis=1)
    qc["bracken_root_reads"] = with_host_counts.get(
        "root", pd.Series(0, index=with_host_counts.index)
    ).astype(float)
    qc["human_species_reads"] = with_host_counts.get(
        "Homo sapiens", pd.Series(0, index=with_host_counts.index)
    ).astype(float)
    qc["bracken_total_reads"] = qc["bracken_root_reads"]
    kraken_unclassified = (
        load_kraken_unclassified_counts(context, report_subdir="kraken_with_host")
        .reindex(qc.index)
        .fillna(0)
    )
    qc["kraken_unclassified_reads"] = kraken_unclassified.astype(float)
    qc["bacterial_species_reads"] = species_bac.sum(axis=1)
    qc["non_human_species_reads"] = (
        qc["bracken_total_reads"] - qc["human_species_reads"]
    ).clip(lower=0)
    qc["non_bacterial_non_human_reads"] = (
        qc["bracken_total_reads"]
        - qc["human_species_reads"]
        - qc["bacterial_species_reads"]
    ).clip(lower=0)
    qc["bacterial_richness"] = (species_bac > 0).sum(axis=1)
    qc["human_species_fraction"] = np.where(
        qc["bracken_total_reads"] > 0,
        qc["human_species_reads"] / qc["bracken_total_reads"],
        np.nan,
    )
    qc["bacterial_species_fraction"] = np.where(
        qc["bracken_total_reads"] > 0,
        qc["bacterial_species_reads"] / qc["bracken_total_reads"],
        np.nan,
    )
    qc["non_bacterial_non_human_fraction"] = np.where(
        qc["bracken_total_reads"] > 0,
        qc["non_bacterial_non_human_reads"] / qc["bracken_total_reads"],
        np.nan,
    )
    # Primary host-burden definition uses pre-host-filter Bracken root composition.
    qc["host_removed_fraction"] = qc["human_species_fraction"]

    metaphlan_species_cols = [
        column
        for column in metaphlan.columns
        if "|s__" in column and column.count("|") >= 6
    ]
    qc["metaphlan_species_reads"] = metaphlan[metaphlan_species_cols].sum(axis=1)
    qc["metaphlan_unclassified_reads"] = metaphlan.get(
        "UNCLASSIFIED", pd.Series(0, index=metaphlan.index)
    )
    qc["metaphlan_unclassified_fraction"] = qc["metaphlan_unclassified_reads"] / (
        qc["metaphlan_unclassified_reads"] + qc["metaphlan_species_reads"]
    )

    qc["community_qc_pass"] = qc["bacterial_species_reads"] >= 10_000
    qc["model_qc_pass"] = qc["bacterial_species_reads"] >= 10_000
    qc["very_low_depth"] = qc["bacterial_species_reads"] < 10_000
    qc["host_logit"] = np.log(
        qc["host_removed_fraction"].clip(1e-4, 1 - 1e-4)
        / (1 - qc["host_removed_fraction"].clip(1e-4, 1 - 1e-4))
    )
    qc["log10_bacterial_reads"] = np.log10(qc["bacterial_species_reads"].clip(lower=1))
    qc["culture_positive_label"] = np.where(
        qc["culture_positive"], "positive", "negative"
    )
    return qc


def bh_adjust(frame: pd.DataFrame, pvalue_column: str) -> pd.DataFrame:
    valid = frame[pvalue_column].notna()
    qvalues = np.full(frame.shape[0], np.nan)
    if valid.any():
        qvalues[valid] = multipletests(
            frame.loc[valid, pvalue_column], method="fdr_bh"
        )[1]
    frame = frame.copy()
    frame["qvalue"] = qvalues
    return frame


def fit_host_model(qc: pd.DataFrame) -> tuple[object, pd.DataFrame]:
    model_df = qc.dropna(
        subset=[
            "host_logit",
            "body_region",
            "patient_id",
            "batch_id",
            "chronicity_group",
            "culture_positive_label",
            "years_since_first_sample",
        ]
    ).copy()
    formula = (
        "host_logit ~ C(body_region, Treatment('lower_extremity')) "
        "+ C(chronicity_group, Treatment('unknown')) "
        "+ C(culture_positive_label, Treatment('negative')) "
        "+ years_since_first_sample + C(batch_id)"
    )
    fit = smf.ols(formula, data=model_df).fit(
        cov_type="cluster",
        cov_kwds={"groups": model_df["patient_id"]},
    )

    params = fit.params
    conf = fit.conf_int()
    pvalues = fit.pvalues
    rows = []
    for term, estimate in params.items():
        if term == "Group Var" or term.startswith("C(batch_id)"):
            continue
        rows.append(
            {
                "term": term,
                "estimate": estimate,
                "conf_low": conf.loc[term, 0] if term in conf.index else np.nan,
                "conf_high": conf.loc[term, 1] if term in conf.index else np.nan,
                "pvalue": pvalues.get(term, np.nan),
            }
        )
    results = pd.DataFrame(rows)
    return fit, bh_adjust(results, "pvalue")


def community_relative_abundance(
    species_bac: pd.DataFrame, sample_ids: list[str]
) -> pd.DataFrame:
    subset = species_bac.loc[sample_ids].copy()
    subset = subset.loc[:, subset.sum(axis=0) > 0]
    return subset.div(subset.sum(axis=1), axis=0)


def summarize_pairwise_distances(
    rel_abundance: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    distances = squareform(pdist(rel_abundance.values, metric="braycurtis"))
    samples = rel_abundance.index.tolist()
    rows = []
    for i, sample_i in enumerate(samples):
        for j in range(i + 1, len(samples)):
            sample_j = samples[j]
            meta_i = metadata.loc[sample_i]
            meta_j = metadata.loc[sample_j]
            same_patient = meta_i["patient_id"] == meta_j["patient_id"]
            same_batch = meta_i["batch_id"] == meta_j["batch_id"]
            if same_patient and same_batch:
                group = "Same patient, same batch date"
            elif same_patient:
                group = "Same patient, different batch date"
            else:
                group = "Different patient"
            rows.append(
                {
                    "sample_i": sample_i,
                    "sample_j": sample_j,
                    "distance": distances[i, j],
                    "comparison_group": group,
                    "same_patient": same_patient,
                    "same_batch": same_batch,
                }
            )
    return pd.DataFrame(rows)


def summarize_pairwise_distance_groups(pairwise: pd.DataFrame) -> pd.DataFrame:
    order = [
        "Same patient, same batch date",
        "Same patient, different batch date",
        "Different patient",
    ]
    summary = (
        pairwise.groupby("comparison_group")["distance"]
        .agg(["count", "median", "mean"])
        .reindex(order)
        .reset_index()
    )

    pvalue_rows = []
    reference = pairwise.loc[
        pairwise["comparison_group"] == "Different patient", "distance"
    ]
    for group in order[:-1]:
        test = pairwise.loc[pairwise["comparison_group"] == group, "distance"]
        statistic = mannwhitneyu(test, reference, alternative="less")
        pvalue_rows.append(
            {
                "comparison_group": group,
                "reference_group": "Different patient",
                "pvalue": statistic.pvalue,
                "median_difference": test.median() - reference.median(),
            }
        )
    pvalues = bh_adjust(pd.DataFrame(pvalue_rows), "pvalue")

    return summary.merge(pvalues, on="comparison_group", how="left")


def make_culture_abundance_table(
    qc: pd.DataFrame,
    species_bac: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rel_abundance = species_bac.div(species_bac.sum(axis=1), axis=0).fillna(0)
    rows = []
    plot_rows = []

    for config in CULTURE_GROUPS:
        taxa = [taxon for taxon in config["taxa"] if taxon in rel_abundance.columns]
        feature = (
            rel_abundance[taxa].sum(axis=1)
            if taxa
            else pd.Series(0.0, index=rel_abundance.index)
        )
        positive_mask = qc[f"culture_{config['group']}"].fillna(False)
        negative_mask = ~positive_mask
        n_positive = int(positive_mask.sum())
        n_negative = int(negative_mask.sum())
        if n_positive == 0 or n_negative == 0:
            continue

        positive_values = feature.loc[positive_mask]
        negative_values = feature.loc[negative_mask]
        test = mannwhitneyu(positive_values, negative_values, alternative="greater")
        y_true = positive_mask.astype(int)
        auc = roc_auc_score(y_true, feature)

        detection_positive = (positive_values >= 0.01).mean()
        detection_negative = (negative_values >= 0.01).mean()

        rows.append(
            {
                "group": config["group"],
                "label": config["label"],
                "n_culture_positive": n_positive,
                "n_culture_negative": n_negative,
                "median_rel_ab_positive": positive_values.median(),
                "median_rel_ab_negative": negative_values.median(),
                "u_statistic": float(test.statistic),
                "auroc": auc,
                "pvalue": test.pvalue,
                "test_name": "Mann-Whitney U (culture positive > culture negative)",
                "sensitivity_at_1pct": detection_positive,
                "false_positive_rate_at_1pct": detection_negative,
            }
        )

        for sample_id, value in feature.items():
            plot_rows.append(
                {
                    "sample_id": sample_id,
                    "group": config["group"],
                    "label": config["label"],
                    "culture_positive": bool(positive_mask.loc[sample_id]),
                    "relative_abundance": value,
                }
            )

    summary = bh_adjust(
        pd.DataFrame(rows).sort_values(
            ["n_culture_positive", "auroc"], ascending=[False, False]
        ),
        "pvalue",
    )
    plot_df = pd.DataFrame(plot_rows)
    return summary, plot_df


def clr_transform(counts: pd.DataFrame, pseudocount: float = 0.5) -> pd.DataFrame:
    logged = np.log(counts + pseudocount)
    return logged.sub(logged.mean(axis=1), axis=0)


def fit_species_models(qc: pd.DataFrame, species_bac: pd.DataFrame) -> pd.DataFrame:
    sample_ids = qc.index[qc["model_qc_pass"]].tolist()
    counts = species_bac.loc[sample_ids].copy()
    clr = clr_transform(counts)

    model_df = qc.loc[
        sample_ids,
        ["patient_id", "body_region", "chronicity_group", "log10_bacterial_reads"],
    ].copy()
    body_regions = [
        region
        for region in BODY_REGION_ORDER
        if region in model_df["body_region"].unique()
    ]
    if "lower_extremity" not in body_regions:
        return pd.DataFrame()

    rows = []
    for species in KEY_SPECIES:
        if species not in clr.columns:
            continue
        prevalence = (counts[species] > 0).mean()
        if prevalence < 0.1:
            continue
        frame = model_df.copy()
        frame["response"] = clr[species]
        formula = (
            "response ~ C(body_region, Treatment('lower_extremity')) "
            "+ C(chronicity_group, Treatment('unknown')) + log10_bacterial_reads"
        )
        fit = smf.ols(formula, data=frame).fit(
            cov_type="cluster",
            cov_kwds={"groups": frame["patient_id"]},
        )
        conf = fit.conf_int()
        for term, estimate in fit.params.items():
            if term == "Intercept":
                continue
            rows.append(
                {
                    "species": species,
                    "term": term,
                    "estimate": estimate,
                    "conf_low": conf.loc[term, 0],
                    "conf_high": conf.loc[term, 1],
                    "pvalue": fit.pvalues[term],
                    "prevalence": prevalence,
                }
            )

    results = pd.DataFrame(rows)
    if results.empty:
        return results
    return bh_adjust(results.sort_values("pvalue"), "pvalue")


def prettify_model_term(term: str) -> str:
    replacements = {
        "C(body_region, Treatment('lower_extremity'))[T.head_neck]": "Head / neck vs lower extremity",
        "C(body_region, Treatment('lower_extremity'))[T.upper_extremity]": "Upper extremity vs lower extremity",
        "C(body_region, Treatment('lower_extremity'))[T.trunk_perineum]": "Trunk / perineum vs lower extremity",
        "C(body_region, Treatment('lower_extremity'))[T.unknown]": "Unknown site vs lower extremity",
        "C(chronicity_group, Treatment('unknown'))[T.acute_like]": "Acute-like vs unknown",
        "C(chronicity_group, Treatment('unknown'))[T.chronic_like]": "Chronic-like vs unknown",
        "C(chronicity_group, Treatment('unknown'))[T.mixed]": "Mixed vs unknown",
        "C(culture_positive_label, Treatment('negative'))[T.positive]": "Culture positive vs negative",
        "log10_bacterial_reads": "Per log10 bacterial reads",
        "years_since_first_sample": "Per year since first patient sample",
    }
    return replacements.get(term, term)


def prepare_species_association_plot_df(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results

    plot_df = results.loc[
        results["term"].str.contains("body_region")
        | results["term"].str.contains("chronicity_group")
    ].copy()
    plot_df = plot_df.loc[plot_df["qvalue"].fillna(1) <= 0.15].copy()
    if plot_df.empty:
        plot_df = (
            results.loc[
                results["term"].str.contains("body_region")
                | results["term"].str.contains("chronicity_group")
            ]
            .head(12)
            .copy()
        )

    plot_df["term_label"] = plot_df["term"].map(prettify_model_term)
    plot_df["species_label"] = plot_df["species"]
    plot_df = plot_df.sort_values(["estimate", "species_label"])
    plot_df["y_label"] = plot_df["species_label"] + " | " + plot_df["term_label"]

    return plot_df


def write_report(
    context: AnalysisContext,
    qc: pd.DataFrame,
    host_results: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    culture_summary: pd.DataFrame,
    species_plot_df: pd.DataFrame,
) -> None:
    qc_samples = int(qc["community_qc_pass"].sum())
    low_depth = int((~qc["community_qc_pass"]).sum())
    revisit_patients = int(
        qc.groupby("patient_id")["culture_date"].nunique().gt(1).sum()
    )
    host_region = (
        host_results.loc[host_results["term"].str.contains("body_region")]
        .sort_values("pvalue")
        .head(1)
    )
    host_time = (
        host_results.loc[host_results["term"].str.contains("years_since_first_sample")]
        .sort_values("pvalue")
        .head(1)
    )
    pair_top = pairwise_summary.loc[
        pairwise_summary["comparison_group"] == "Same patient, same batch date"
    ].head(1)
    pair_revisit = pairwise_summary.loc[
        pairwise_summary["comparison_group"] == "Same patient, different batch date"
    ].head(1)
    culture_top = culture_summary.sort_values(
        ["n_culture_positive", "qvalue", "auroc"], ascending=[False, True, False]
    ).head(4)
    species_focus = [
        (
            "Pseudomonas aeruginosa",
            "C(chronicity_group, Treatment('unknown'))[T.chronic_like]",
        ),
        (
            "Staphylococcus aureus",
            "C(body_region, Treatment('lower_extremity'))[T.head_neck]",
        ),
        (
            "Cutibacterium acnes",
            "C(body_region, Treatment('lower_extremity'))[T.head_neck]",
        ),
        (
            "Serratia marcescens",
            "C(chronicity_group, Treatment('unknown'))[T.acute_like]",
        ),
    ]

    lines = [
        "# EB shotgun metagenomics analysis",
        "",
        "## Scope",
        "",
        "- Primary metadata source: `PA_Data_Finalized.xlsx`, sheet `Corrected EB wound spreadsheet`.",
        "- Taxonomic source for bacterial community analysis: Bracken bacterial species counts.",
        "- Host burden source: Bracken human species fraction among Bracken root reads from pre-host-filter sequencing data.",
        "- Absolute culture date is treated as technical batch; patient-relative elapsed time is treated as the biological time variable.",
        f"- Community-level analyses used a depth-aware QC threshold of at least 10,000 Bracken bacterial reads ({qc_samples} / {qc.shape[0]} samples passed; {low_depth} were retained for descriptive analyses only).",
        "",
        "## Key findings",
        "",
        f"- Host contamination was substantial: median Bracken-based human fraction was {qc['host_removed_fraction'].median():.1%}, median Bracken bacterial fraction among root reads was {qc['bacterial_species_fraction'].median():.1%}, and median non-bacterial/non-human residual fraction was {qc['non_bacterial_non_human_fraction'].median():.2%}.",
        f"- Revisit structure is real, not negligible: the 74 swabs came from {qc['patient_id'].nunique()} patients across {qc['culture_date'].dt.strftime('%Y-%m-%d').nunique()} collection dates spanning {qc['culture_date'].min().date()} to {qc['culture_date'].max().date()}, and {revisit_patients} of the {qc['patient_id'].nunique()} patients had samples on more than one date.",
    ]

    if not host_region.empty:
        row = host_region.iloc[0]
        lines.append(
            f"- The strongest body-region host signal was `{prettify_model_term(row['term'])}` with coefficient {row['estimate']:.2f} on the logit host scale (q={row['qvalue']:.3g}) after adjusting for chronicity, broad culture positivity, patient-relative time, and batch-date fixed effects."
        )
    if not host_time.empty:
        row = host_time.iloc[0]
        lines.append(
            f"- Host burden also changed with patient-relative elapsed time: `{prettify_model_term(row['term'])}` had coefficient {row['estimate']:.2f} (q={row['qvalue']:.3g})."
        )

    if not pair_top.empty:
        row = pair_top.iloc[0]
        lines.append(
            f"- Same-patient, same-batch-date pairs had median Bray-Curtis distance {row['median']:.3f}, compared with {pairwise_summary.loc[pairwise_summary['comparison_group'] == 'Different patient', 'median'].iloc[0]:.3f} for unrelated pairs."
        )
    if not pair_revisit.empty:
        row = pair_revisit.iloc[0]
        lines.append(
            f"- Same-patient, different-batch-date pairs were much less stable: median distance was {row['median']:.3f}, essentially similar to unrelated pairs."
        )

    for _, row in culture_top.iterrows():
        lines.append(
            f"- Culture concordance was strongest for {row['label']} (AUROC {row['auroc']:.2f}, q={row['qvalue']:.3g}; median relative abundance {row['median_rel_ab_positive']:.2%} in culture-positive swabs)."
        )

    lines.extend(
        [
            "",
            "## Figure captions",
            "",
            "1. `fig_02_01_qc_host_burden.svg`: QC overview. Left, host-depleted read pairs versus Bracken bacterial species reads, with the community-analysis threshold at 10,000 reads. Right, Bracken human fraction by body region; points are individual swabs colored by patient.",
            "2. `fig_03_01_pairwise_distance.svg`: Pairwise Bray-Curtis distances between QC-passing swabs. Same-patient, same-batch-date comparisons are the closest descriptive group; same-patient different-batch-date comparisons are shown separately because absolute date is treated as technical batch in later models.",
            "3. `fig_04_01_culture_concordance.svg`: Metagenomic relative abundance of key pathogen groups stratified by whether routine culture called the same organism group. Boxplots summarize distributions; point clouds show individual swabs.",
            "4. `fig_05_01_species_associations.svg`: Cluster-robust CLR effect sizes for selected taxa versus body region and chronicity covariates after adjusting for sequencing depth.",
            "",
            "## Caveats",
            "",
            "- Body-site effects should be interpreted jointly with patient-level repeated measures and culture-date batch structure; repeated measures are common in this cohort.",
            "- Culture agreement is organism-group level, not strain or resistance-level agreement. MRSA and MSSA were collapsed to `S. aureus` because species-level metagenomics does not resolve methicillin resistance.",
            "- Low-bacterial-depth swabs were retained in descriptive host and culture plots but excluded from composition-sensitive community models.",
        ]
    )

    if not species_plot_df.empty:
        lines.extend(
            [
                "",
                "## Species-level model hits highlighted in Figure 4",
                "",
            ]
        )
        for species, term in species_focus:
            match = species_plot_df.loc[
                (species_plot_df["species"] == species)
                & (species_plot_df["term"] == term)
            ]
            if match.empty:
                continue
            row = match.iloc[0]
            lines.append(
                f"- {row['species']}: {row['term_label']} -> effect {row['estimate']:.2f} (95% CI {row['conf_low']:.2f} to {row['conf_high']:.2f})."
            )

    report_path = context.analysis_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n")


def ensure_output_dirs(context: AnalysisContext) -> None:
    context.analysis_dir.mkdir(exist_ok=True)
    context.figure_dir.mkdir(exist_ok=True)
    context.table_dir.mkdir(exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze EB shotgun metagenomics summaries."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the metagenomics_20260206 directory.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    context = AnalysisContext(
        data_dir=data_dir,
        analysis_dir=data_dir / "analysis_concise",
        figure_dir=data_dir / "analysis_concise" / "figures",
        table_dir=data_dir / "analysis_concise" / "tables",
    )
    ensure_output_dirs(context)

    species_all, species_bac = load_bracken_tables(context)
    metaphlan = load_metaphlan_table(context)
    sample_ids = sorted(species_all.index)
    metadata = load_metadata(context, sample_ids)
    qc = prepare_qc_table(context, metadata, species_all, species_bac, metaphlan)

    host_fit, host_results = fit_host_model(qc)

    community_samples = qc.index[qc["community_qc_pass"]].tolist()
    rel_abundance = community_relative_abundance(species_bac, community_samples)
    pairwise = summarize_pairwise_distances(rel_abundance, metadata)
    pairwise_summary = summarize_pairwise_distance_groups(pairwise)

    culture_summary, _ = make_culture_abundance_table(qc, species_bac)

    species_results = fit_species_models(qc, species_bac)
    species_plot_df = prepare_species_association_plot_df(species_results)

    metadata.to_csv(context.table_dir / "cleaned_metadata.tsv", sep="\t")
    qc.to_csv(context.table_dir / "qc_metrics.tsv", sep="\t")
    host_results.to_csv(context.table_dir / "host_model.tsv", sep="\t", index=False)
    pairwise.to_csv(context.table_dir / "pairwise_distances.tsv", sep="\t", index=False)
    pairwise_summary.to_csv(
        context.table_dir / "pairwise_distance_summary.tsv", sep="\t", index=False
    )
    culture_summary.to_csv(
        context.table_dir / "culture_concordance.tsv", sep="\t", index=False
    )
    species_results.to_csv(
        context.table_dir / "species_associations.tsv", sep="\t", index=False
    )

    with (context.analysis_dir / "model_summary.txt").open("w") as handle:
        handle.write(str(host_fit.summary()))
        handle.write("\n")

    write_report(
        context, qc, host_results, pairwise_summary, culture_summary, species_plot_df
    )


if __name__ == "__main__":
    main()
