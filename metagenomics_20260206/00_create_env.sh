mamba create -n metagenomics python=3.12 -y

conda activate metagenomics

packages=(
    awscli
    rsync
    pigz
    pbzip2
    fastqc
    fastp
    seqkit
    bwa-mem2
    samtools
    kraken2
    bracken
    tqdm
    bowtie2
    metaphlan
    spades
    bakta
)

mamba install -y -c conda-forge -c bioconda "${packages[@]}"

joined_packages=$(IFS='|'; echo "${packages[*]}")
mamba list "^(${joined_packages})$"

#   Name       Version  Build             Channel    
# ─────────────────────────────────────────────────────
#   awscli     2.33.8   py312h20c3967_0   conda-forge
#   bakta      1.12.0   pyhdfd78af_0      bioconda   
#   bowtie2    2.5.4    he96a11b_7        bioconda   
#   bracken    3.1      h9948957_0        bioconda   
#   bwa-mem2   2.3      he70b90d_0        bioconda   
#   fastp      1.1.0    heae3180_0        bioconda   
#   fastqc     0.12.1   hdfd78af_0        bioconda   
#   kraken2    2.17.1   pl5321h077b44d_0  bioconda   
#   metaphlan  4.2.4    pyhdfd78af_0      bioconda   
#   pbzip2     1.1.13   h1fcc475_2        conda-forge
#   pigz       2.8      h421ea60_2        conda-forge
#   rsync      3.4.1    h81c0278_2        conda-forge
#   samtools   1.23     h96c455f_0        bioconda   
#   seqkit     2.12.0   he881be0_1        bioconda   
#   spades     4.2.0    h8d6e82b_2        bioconda   
#   tqdm       4.67.3   pyh8f84b5b_0      conda-forge