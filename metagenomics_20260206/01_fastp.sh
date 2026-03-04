#!/bin/bash

# Define directories
DATA_DIR="$HOME/dev/20250320_eb_summary/metagenomics_20260206"

# Create output subdirectories
mkdir -p "$DATA_DIR/fastqc_before"
mkdir -p "$DATA_DIR/fastqc_after"
mkdir -p "$DATA_DIR/fastp"

seqkit stats -j 8 $DATA_DIR/fastq/*.fastq.gz > "$DATA_DIR/input.stats"

# Loop over all paired-end files and process sequentially
i=0
for READ1 in $DATA_DIR/fastq/*_R1_001.fastq.gz; do
    i=$((i + 1))
    # skip the first 41 samples
    # if (( i <= 41 )); then
    #     echo "Skipping sample $i: $READ1"
    #     continue
    # fi
    # Derive READ2 and sample name
    READ2="${READ1%_R1_001.fastq.gz}_R2_001.fastq.gz"
    BASENAME=$(basename "$READ1")
    SAMPLE_NAME="${BASENAME%%_*}"

    echo "Processing sample: $SAMPLE_NAME"

    # Step 1: FastQC before fastp
    echo "Running FastQC before fastp for $SAMPLE_NAME..."
    fastqc -o "$DATA_DIR/fastqc_before" "$READ1" "$READ2"

    # Step 2: Run fastp for adapter trimming and quality filtering
    echo "Running fastp for $SAMPLE_NAME..."
    fastp \
        -i "$READ1" \
        -I "$READ2" \
        -o "$DATA_DIR/fastp/${SAMPLE_NAME}_1.fq.gz" \
        -O "$DATA_DIR/fastp/${SAMPLE_NAME}_2.fq.gz" \
        --cut_tail \
        --html "$DATA_DIR/fastp/${SAMPLE_NAME}.html" \
        --json "$DATA_DIR/fastp/${SAMPLE_NAME}.json"

    # Step 3: FastQC after fastp
    echo "Running FastQC after fastp for $SAMPLE_NAME..."
    fastqc -o "$DATA_DIR/fastqc_after" \
        "$DATA_DIR/fastp/${SAMPLE_NAME}_1.fq.gz" \
        "$DATA_DIR/fastp/${SAMPLE_NAME}_2.fq.gz"

    # Remoev the input fastq files to save space
    rm "$READ1" "$READ2"
    echo "Sample $SAMPLE_NAME processing complete."
done

seqkit stats -j 8 $DATA_DIR/fastp/*.fq.gz > "$DATA_DIR/fastp.stats"