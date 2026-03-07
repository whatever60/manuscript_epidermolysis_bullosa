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
# # 01. Project Inventory And Metadata
#
# This notebook establishes the analysis inputs, harmonizes sample identifiers, merges the two metadata workbooks,
# and defines the patient-date, batch, and patient-relative time variables used later in the modeling workflow.
#

# %%
from pathlib import Path
import sys

import pandas as pd
from IPython.display import Markdown, SVG, display

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import workflow_core as wc

context, base_data, base, advanced = wc.bootstrap_notebook()


# %% [markdown]
# ## Define Notebook-Local Metadata Helpers
#
# These helper functions are kept inside this notebook because they are only used here.
#


# %%
def metadata_export_frame(metadata):
    return metadata.reset_index().rename(columns={"index": "sample_id"})


def load_full_primary_metadata():
    primary = pd.read_excel(
        context.data_dir / "metadata" / "PA_Data_Finalized.xlsx",
        sheet_name="Corrected EB wound spreadsheet",
    ).rename(
        columns={
            "Code": "code",
            "Date of Culture": "culture_date",
            "Location": "location_raw",
            "Clinical Correlates (chronicity, clinical signs of infection, SCC)": "clinical_correlates",
            "Result": "culture_result",
            "Gram Stain": "gram_stain",
            "Sent to…": "sent_to",
        }
    )
    primary["sample_id"] = primary["code"].map(base.normalize_sample_id)
    primary = primary.loc[primary["sample_id"].astype(str).str.strip() != ""].copy()
    primary["patient_id"] = primary["sample_id"].str.slice(0, 2)
    primary = primary.loc[
        primary["patient_id"].str.fullmatch(r"\d{2}", na=False)
    ].copy()
    primary["culture_date"] = pd.to_datetime(primary["culture_date"], errors="coerce")
    primary = primary.loc[primary["culture_date"].notna()].copy()
    primary["location"] = primary["location_raw"].map(base.standardize_location)
    primary["body_region"] = primary["location"].map(base.infer_body_region)
    primary["laterality"] = primary["location"].map(base.infer_laterality)
    primary["visit_id"] = (
        primary["patient_id"] + "_" + primary["culture_date"].dt.strftime("%Y-%m-%d")
    )
    return primary


def full_metadata_site_visit_summary():
    primary = load_full_primary_metadata()
    per_patient = (
        primary.groupby("patient_id")
        .agg(
            n_records=("sample_id", "size"),
            n_dates=("culture_date", "nunique"),
            n_visits=("visit_id", "nunique"),
            n_clean_locations=("location", "nunique"),
            n_body_regions=("body_region", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["n_records", "n_clean_locations", "n_dates"],
            ascending=[False, False, False],
        )
    )
    visit_sites = (
        primary.groupby(["patient_id", "visit_id"])["location"]
        .apply(lambda s: sorted(set(s)))
        .reset_index(name="sites")
    )
    visit_sites["n_sites"] = visit_sites["sites"].str.len()
    revisited_sites = (
        primary.groupby(["patient_id", "location"])["visit_id"]
        .apply(lambda s: sorted(set(s)))
        .reset_index(name="visits")
    )
    revisited_sites["n_visits"] = revisited_sites["visits"].str.len()
    overall = pd.DataFrame(
        [
            {
                "n_rows": int(primary.shape[0]),
                "n_patients": int(primary["patient_id"].nunique()),
                "n_patient_visit_pairs": int(
                    primary[["patient_id", "visit_id"]].drop_duplicates().shape[0]
                ),
                "n_patient_site_pairs": int(
                    primary[["patient_id", "location"]].drop_duplicates().shape[0]
                ),
                "multisite_visits": int((visit_sites["n_sites"] > 1).sum()),
                "total_visits": int(visit_sites.shape[0]),
                "revisited_sites": int((revisited_sites["n_visits"] > 1).sum()),
                "total_patient_sites": int(revisited_sites.shape[0]),
                "median_dates_per_patient": float(per_patient["n_dates"].median()),
                "median_sites_per_patient": float(
                    per_patient["n_clean_locations"].median()
                ),
            }
        ]
    )
    return {
        "overall": overall,
        "per_patient": per_patient,
        "visit_sites": visit_sites,
        "revisited_sites": revisited_sites,
        "primary": primary,
    }


def repeated_measure_summary(metadata):
    grouped = metadata.reset_index().groupby("patient_id")
    summary = grouped.agg(
        n_samples=("sample_id", "size"),
        n_visits=("visit_id", "nunique"),
        n_dates=("culture_date", "nunique"),
    ).reset_index()
    summary["multi_date"] = summary["n_dates"] > 1
    return summary.sort_values(
        ["n_dates", "n_samples", "patient_id"], ascending=[False, False, True]
    )


# %% [markdown]
# ## Load And Save The Cleaned Metadata Table
#
# The primary metadata source is `PA_Data_Finalized.xlsx`, sheet `Corrected EB wound spreadsheet`.
# Missing date, location, and culture-note fields are backfilled from the lab-archive workbook.
#

# %%
metadata = base_data["metadata"]
metadata_table = metadata_export_frame(metadata)
wc.save_table(metadata_table, wc.table_path(context, 1, "cleaned_metadata"))

display(
    metadata_table[
        [
            "sample_id",
            "patient_id",
            "visit_id",
            "batch_id",
            "culture_date",
            "days_since_first_sample",
            "years_since_first_sample",
            "location_raw",
            "location",
            "body_region",
            "laterality",
            "chronicity_group",
            "clinical_infection_flag",
            "culture_result",
        ]
    ].head(20)
)


# %% [markdown]
# ## Patient-Date, Batch, And Elapsed-Time Structure
#
# `visit_id` is still defined as `patient_id + culture_date` so same-day multisite swabs stay grouped.
# But later models now separate two different roles of the date:
# the absolute `culture_date` is treated as a technical batch variable, while elapsed time since a patient's first sample
# is treated as the biological time variable.
#

# %% [markdown]
# ## Metadata Processing Logic
#
# The metadata-cleaning logic is intentionally explicit:
#
# - `culture_date` is preserved in three derived forms.
# - `visit_id = patient_id + culture_date` is retained only as a patient-date identifier for same-day multisite grouping.
# - `batch_id = culture_date` is the technical batch/date variable used in updated host, similarity, and concordance models.
# - `days_since_first_sample` and `years_since_first_sample` are the patient-relative biological time variables used instead of absolute year for longitudinal interpretation.
# - Therefore, later notebooks do **not** interpret absolute date as biology by default.
# - `patient + site` is not equivalent to `patient + visit_id` in this cohort: some dates contain multiple sampled body sites, and some body sites are revisited across different dates.
# - Raw `Location`, `Clinical Correlates`, `Result`, and `Gram Stain` are preserved and then mapped into cleaned analysis variables so the transformation remains auditable.
#

# %%
visit_summary = repeated_measure_summary(metadata)
full_meta = full_metadata_site_visit_summary()
full_overall = full_meta["overall"].iloc[0]
multisite_examples = (
    full_meta["visit_sites"].loc[full_meta["visit_sites"]["n_sites"] > 1].head(10)
)
revisited_examples = (
    full_meta["revisited_sites"]
    .loc[full_meta["revisited_sites"]["n_visits"] > 1]
    .head(10)
)
body_region_counts = (
    metadata_table["body_region"]
    .value_counts(dropna=False)
    .rename_axis("body_region")
    .reset_index(name="n_samples")
)
chronicity_counts = (
    metadata_table["chronicity_group"]
    .value_counts(dropna=False)
    .rename_axis("chronicity_group")
    .reset_index(name="n_samples")
)

display(body_region_counts)
display(chronicity_counts)
display(visit_summary)
display(full_meta["overall"])
display(multisite_examples)
display(revisited_examples)

summary_lines = [
    f"- Sequenced swabs: {len(metadata_table)} from {metadata_table['patient_id'].nunique()} patients.",
    f"- Unique patient-date identifiers: {metadata_table['visit_id'].nunique()} across {metadata_table['batch_id'].nunique()} culture-date batches.",
    f"- Patients sampled on more than one date: {int(visit_summary['multi_date'].sum())} of {visit_summary.shape[0]}.",
    f"- In the full primary metadata sheet, valid coded rows covered {int(full_overall['n_rows'])} samples from {int(full_overall['n_patients'])} patients.",
    f"- Full-metadata patient-date pairs: {int(full_overall['n_patient_visit_pairs'])}; patient-site pairs: {int(full_overall['n_patient_site_pairs'])}.",
    f"- Multisite same-date sampling occurred in {int(full_overall['multisite_visits'])} of {int(full_overall['total_visits'])} patient-date groups.",
    f"- Revisited patient-site combinations occurred in {int(full_overall['revisited_sites'])} of {int(full_overall['total_patient_sites'])} patient-site groups.",
    f"- Median patient-relative elapsed time was {metadata_table['days_since_first_sample'].median():.0f} days since first sample.",
    "- Cleaned body-site labels were collapsed into head/neck, upper extremity, trunk/perineum, and lower extremity.",
    "- Clinical free text was parsed into chronicity and infection-status categories; these are rule-based and remain approximate.",
]
display(Markdown("## Working Summary\n" + "\n".join(summary_lines)))
