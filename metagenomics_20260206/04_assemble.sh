THREADS=16
BAKTA_DB="$HOME/data/bakta_db/db"
DATA_DIR="$HOME/dev/20250320_eb_summary/metagenomics_20260206"

mkdir -p "$DATA_DIR/spades"
mkdir -p "$DATA_DIR/assembly"
mkdir -p "$DATA_DIR/prokka"

i=0
for READ1 in "$DATA_DIR/fastp_no_host/"*_1.fq.gz; do
    
    # skip the first 39 files
    ((i++))
    # if [[ $i -lt 40 ]]; then
    #     continue
    # fi

    READ2="${READ1%_1.fq.gz}_2.fq.gz"
    BASENAME=$(basename "$READ1")
    SAMPLE_NAME="${BASENAME%_1.fq.gz}"

    if [[ ! -f "$READ2" ]]; then
        echo "Warning: No matching R2 file found for $READ1. Skipping..."
        continue
    fi

    echo "Running SPAdes for $SAMPLE_NAME"
    SAMPLE_OUT="$DATA_DIR/spades/${SAMPLE_NAME}_assembly"
    mkdir -p "$SAMPLE_OUT"

    spades.py \
        --meta \
        -1 "$READ1" \
        -2 "$READ2" \
        -o "$SAMPLE_OUT" \
        --threads $THREADS > /dev/null

    cp "$SAMPLE_OUT/contigs.fasta" "$DATA_DIR/assembly/${SAMPLE_NAME}.fna"

    # echo "Running Prokka for $SAMPLE_NAME"
    # PROKKA_OUT="$DATA_DIR/prokka/${SAMPLE_NAME}"
    # mkdir -p "$PROKKA_OUT"
    # prokka \
    #     --rnammer \
    #     --rfam \
    #     --cpus $THREADS \
    #     --outdir "$PROKKA_OUT" \
    #     --prefix "$SAMPLE_NAME" \
    #     --force \
    #     --centre X --compliant \
    #     "$DATA_DIR/assembly/${SAMPLE_NAME}.fna"
    # prokka \
    #     --cpus $THREADS \
    #     --outdir "$PROKKA_OUT" \
    #     --prefix "$SAMPLE_NAME" \
    #     --metagenome \
    #     --mincontiglen 200 \
    #     --centre X \
    #     --compliant \
    #     --force \
    #     "$DATA_DIR/assembly/${SAMPLE_NAME}.fna"

    # echo "Running Bakta for $SAMPLE_NAME"
    # BAKTA_OUT="$DATA_DIR/bakta/${SAMPLE_NAME}"
    # mkdir -p "$BAKTA_OUT"
    # bakta \
    #     --db "$BAKTA_DB" \
    #     --threads "$THREADS" \
    #     --output "$BAKTA_OUT" \
    #     --prefix "$SAMPLE_NAME" \
    #     --meta \
    #     --min-contig-length 200 \
    #     --compliant \
    #     --force \
    #     "$DATA_DIR/assembly/${SAMPLE_NAME}.fna"
done
