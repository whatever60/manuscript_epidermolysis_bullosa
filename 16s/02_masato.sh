#!/usr/bin/env bash
set -euo pipefail

# pip install masato==1.5.3

# --- Conda env activation ---
ENV_NAME="masato"
CONDA_EXE=$(command -v conda)
if [ -z "$CONDA_EXE" ]; then
    echo "Error: 'conda' command not found in PATH." >&2
    exit 1
fi
__conda_setup="$("$CONDA_EXE" "shell.bash" "hook" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    echo "Error: Failed to initialize conda shell." >&2
    exit 1
fi
unset __conda_setup
echo "Activating conda environment: $ENV_NAME"
set +u  # somehow the $JAVA_LD_LIBRARY_PATH var is not set during conda activate
conda activate "$ENV_NAME"
set -u


data_dir="./output_unoise3_16s"
cores=16
preprocess.py combine-trim-merge-pe \
    --input-dir ./fastq \
    --output $data_dir/merged.fq.gz \
    --min-length 200 \
    --cores $cores

usearch_workflow.py workflow_per_sample \
    --input_fastq $data_dir/merged.fq.gz \
    --output_path $data_dir/unoise3_zotu.json \
    --search \
    --num_threads $cores

usearch_workflow.py aggregate_samples \
    --input_json $data_dir/unoise3_zotu.json \
    --output_fasta $data_dir/unoise3_zotu.fa \
    --output_count $data_dir/unoise3_zotu.biom \
    --prefix 16S-U

run_rdp_classifier.py -i $data_dir/unoise3_zotu.fa -d 16srrna
get_tree.py -i $data_dir/unoise3_zotu.fa
