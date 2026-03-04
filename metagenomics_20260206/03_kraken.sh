#!/usr/bin/env bash

# NOTE: this script requires sudo. Otherwise, run sudo mount -o remount,size=128G /dev/shm 
# with sudo and then this script without sudo.

# Exit immediately if a command exits with a non-zero status.
# Treat unset variables as an error when substituting.
# The exit status of a pipeline is that of the last command to exit with a non-zero status.
set -euo pipefail

#== Configuration ==#
DATA_DIR="$HOME/dev/20250320_eb_summary/metagenomics_20260206"
KRAKEN_DB="$HOME/data/kraken2_db/standard_clean"      # Kraken2 DB on disk
KRAKEN_DB_MEM=/dev/shm/kraken2_db/standard_clean # Kraken2 DB in memory
THREADS=16

#== Cleanup Function ==#
# This function is called on exit to remove the database from the RAM disk.
cleanup() {
    echo " " # Newline for cleaner exit
    echo "🧹 Cleaning up: Removing Kraken2 database from RAM disk..."
    # Check if the directory exists before trying to remove it to avoid errors.
    if [ -d "$KRAKEN_DB_MEM" ]; then
        rm -rf "$KRAKEN_DB_MEM"
        echo "✅ Cleanup complete."
    else
        echo "ℹ️ In-memory database directory not found, skipping removal."
    fi
}
# Trap signals to ensure cleanup happens even if the script is interrupted or exits normally.
trap cleanup EXIT INT TERM

#== Database Preparation ==#
# Copy the database to the RAM disk before starting the main analysis.
echo "🚀 Preparing for analysis by copying Kraken2 DB to RAM disk..."
echo "   Source: $KRAKEN_DB"
echo "   Destination: $KRAKEN_DB_MEM"
# Ensure the parent directory exists in /dev/shm
mkdir -p "$(dirname "$KRAKEN_DB_MEM")"
# Use rsync for efficient copying; it will skip if files already exist and match.
sudo mount -o remount,size=128G /dev/shm
rsync -ah --info=progress2 --ignore-existing \
    --include='hash.k2d' \
    --include='taxo.k2d' \
    --include='opts.k2d' \
    --include='unmapped.txt' \
    --exclude='*' \
    "$KRAKEN_DB/" "$KRAKEN_DB_MEM/"
echo "✅ Database is ready in memory."
echo "--------------------------------------------------"

#== Main Processing Loop ==#
# Create the output directory for Kraken2 and Bracken results.
# mkdir -p "$DATA_DIR/kraken_with_host"
# fqr1s=$(ls "$DATA_DIR/fastp/"*_1.fq.gz)
# i=0
# for READ1 in $fqr1s; do
#     ((++i))
#     READ2="${READ1%_1.fq.gz}_2.fq.gz"
#     BASENAME=$(basename "$READ1")
#     SAMPLE="${BASENAME%_1.fq.gz}"

#     # This echo will be captured by tqdm and used to update the progress bar description.
#     echo "Processing sample: $SAMPLE"

#     # Run Kraken2 on raw reads using the in-memory database for speed.
#     kraken2 \
#         --db "$KRAKEN_DB_MEM" \
#         --paired \
#         --threads "$THREADS" \
#         --output "$DATA_DIR/kraken_with_host/${SAMPLE}_kraken_output.txt" \
#         --report "$DATA_DIR/kraken_with_host/${SAMPLE}_kraken_report.txt" \
#         --use-names \
#         --memory-mapping \
#         "$READ1" "$READ2" &> /dev/null

#     # Run Bracken on the Kraken2 report to estimate species abundance.
#     # Bracken uses the on-disk database path.
#     bracken \
#         -d "$KRAKEN_DB" \
#         -i "$DATA_DIR/kraken_with_host/${SAMPLE}_kraken_report.txt" \
#         -o "$DATA_DIR/kraken_with_host/${SAMPLE}_bracken_report.txt" \
#         -r 150 \
#         -l S &> /dev/null
#     pigz -p "$THREADS" -f "$DATA_DIR/kraken_with_host/${SAMPLE}_kraken_output.txt"
# done | tqdm --total "$(echo "$fqr1s" | wc -l)" > /dev/null

mkdir -p "$DATA_DIR/kraken"
fqr1s=$(ls "$DATA_DIR/fastp_no_host/"*_1.fq.gz)
i=0
for READ1 in $fqr1s; do
    ((++i))
    READ2="${READ1%_1.fq.gz}_2.fq.gz"
    BASENAME=$(basename "$READ1")
    SAMPLE="${BASENAME%_1.fq.gz}"

    echo "Processing sample: $SAMPLE"

    # Run Kraken2 on reads without host contamination.
    kraken2 \
        --db "$KRAKEN_DB_MEM" \
        --paired \
        --threads "$THREADS" \
        --output "$DATA_DIR/kraken/${SAMPLE}_kraken_output.txt" \
        --report "$DATA_DIR/kraken/${SAMPLE}_kraken_report.txt" \
        --use-names \
        --memory-mapping \
        "$READ1" "$READ2" &> /dev/null

    # Run Bracken on the Kraken2 report for non-host reads.
    # Bracken uses the on-disk database path.
    bracken \
        -d "$KRAKEN_DB" \
        -i "$DATA_DIR/kraken/${SAMPLE}_kraken_report.txt" \
        -o "$DATA_DIR/kraken/${SAMPLE}_bracken_report.txt" \
        -r 150 \
        -l S &> /dev/null
    pigz -p "$THREADS" -f "$DATA_DIR/kraken/${SAMPLE}_kraken_output.txt"
done | tqdm --total "$(echo "$fqr1s" | wc -l)" > /dev/null

# The cleanup function is called automatically here upon normal script completion
# because of the 'trap cleanup EXIT' command set at the beginning.