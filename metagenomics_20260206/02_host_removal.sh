#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# constants
DATA_DIR="$HOME/dev/20250320_eb_summary/metagenomics_20260206"
# BWA-MEM2 index is built on the gzipped FASTA.
REF_GENOME_GZ="$HOME/data/gencode/Gencode_human/release_46/GRCh38.primary_assembly.genome.fa.gz"
# FAI is built on the bgzipped FASTA.
REF_GENOME_BG="${REF_GENOME_GZ%.gz}.bgz"
THREADS=16

fastp_dir="$DATA_DIR/fastp"
map_dir="$DATA_DIR/map_host"           # will store FILTERED (unmapped-pairs-only) CRAM
no_host_dir="$DATA_DIR/fastp_no_host"  # unmapped-pairs FASTQ

mkdir -p "$map_dir" "$no_host_dir"

# Ensure FASTA index exists (needed for CRAM decode)
if [[ ! -f "${REF_GENOME_BG}.fai" ]]; then
    echo "Indexing reference with samtools faidx..."
    samtools faidx "$REF_GENOME_BG"
else
    echo "Reference index found: ${REF_GENOME_BG}.fai"
fi

# Index the reference genome for bwa-mem2 if needed
# NOTE: bwa-mem2 can use gzipped FASTA directly.
idx_files=( "${REF_GENOME_GZ}".bwt.2bit.64 )
if (( ${#idx_files[@]} == 0 )); then
    echo "Indexing the reference genome with bwa-mem2..."
    bwa-mem2 index "$REF_GENOME_GZ"
else
    echo "BWA-MEM2 index files found for ${REF_GENOME_GZ}"
fi

read1_files=( "$fastp_dir/"*_1.fq.gz )
if (( ${#read1_files[@]} == 0 )); then
    echo "No input files found: ${fastp_dir}/*_1.fq.gz" >&2
    exit 1
fi

for read1 in "${read1_files[@]}"; do
    read2="${read1%_1.fq.gz}_2.fq.gz"
    if [[ ! -f "$read2" ]]; then
        echo "Missing mate for: $read1" >&2
        continue
    fi

    basename="$(basename "$read1")"
    sample="${basename%_1.fq.gz}"

    cram_tmp="$map_dir/${sample}.host_all.tmp.cram"
    cram_out="$map_dir/${sample}.cram"  # final: unmapped-pairs-only CRAM

    unmapped1="$no_host_dir/${sample}_1.fq.gz"
    unmapped2="$no_host_dir/${sample}_2.fq.gz"

    echo "[$sample] Aligning to host and writing temporary CRAM..."
    bwa-mem2 mem -t "$THREADS" "$REF_GENOME_GZ" "$read1" "$read2" | \
        samtools view -@ "$THREADS" -C -T "$REF_GENOME_BG" -o "$cram_tmp" -

    echo "[$sample] Extracting unmapped read pairs to FASTQ..."
    samtools fastq -@ "$THREADS" -T "$REF_GENOME_BG" \
        -f 12 -F 256 "$cram_tmp" \
        -1 "$unmapped1" -2 "$unmapped2" \
        -0 /dev/null -s /dev/null -n

    echo "[$sample] Writing filtered CRAM (NOT(both unmapped)) and removing temp CRAM..."
    samtools view -@ "$THREADS" -C -T "$REF_GENOME_BG" \
        -G 12 \
        -o "$cram_out" "$cram_tmp"
    rm -f "$cram_tmp"
done

seqkit stats -j 8 "$no_host_dir/"*.fq.gz > "$DATA_DIR/fastp_no_host.stats"
