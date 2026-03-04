aws s3 sync s3://yiming-qu/20241109_novogene_demux_mouse_rad_exfo_seq_eb_wgs/fastq/ fastq --exclude "*" --include "SUB*"

aws s3 sync s3://yiming-qu/20260206_novogene/01.RawData/ fastq_temp --exclude "*" --include "*meta*.fq.gz"
# Move ./fastq_temp/*/*.fastq.gz to ./fastq and rename to {basename}.split("_")[0]_R1_001.fastq.gz when basename ends with _1.fq.gz, and similarly for _2.fq.gz
mkdir -p fastq
for file in fastq_temp/*/*.fq.gz; do
    filename=$(basename "$file")
    if [[ $filename == *_1.fq.gz ]]; then
        filename_split=(${filename//_/ })
        newname="${filename_split[0]}_R1_001.fastq.gz"
    elif [[ $filename == *_2.fq.gz ]]; then
        filename_split=(${filename//_/ })
        newname="${filename_split[0]}_R2_001.fastq.gz"
    else
        continue
    fi
    mv "$file" "fastq/$newname"
done
rm -rf fastq_temp