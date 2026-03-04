#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"

mamba create -y -n eb -c conda-forge -c bioconda \
    python=3.12 \
    pandas \
    numpy \
    scipy \
    statsmodels \
    scikit-learn \
    seaborn \
    matplotlib \
    openpyxl \
    jupyterlab \
    ipykernel \
    r-base=4.4 \
    r-irkernel \
    r-tidyverse \
    r-vegan \
    r-lme4 \
    r-lmertest \
    r-broom \
    r-broom.mixed \
    r-emmeans

conda activate eb

python -m ipykernel install --user --name eb --display-name "Python (eb)"
R -q -e "IRkernel::installspec(name = 'ir-eb', displayname = 'R (eb)')"
