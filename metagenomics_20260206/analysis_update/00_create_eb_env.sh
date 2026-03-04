#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"

ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eb_environment.yml"
CONDA_BASE="$(dirname "$(dirname "$(command -v conda)")")"
ENV_PREFIX="${CONDA_BASE}/envs/eb"

if [[ -d "${ENV_PREFIX}" ]]; then
    CONDA_NO_PLUGINS=true mamba env update -y -n eb -f "${ENV_FILE}" --prune
else
    CONDA_NO_PLUGINS=true mamba env create -y -f "${ENV_FILE}"
fi

conda activate eb

python -m ipykernel install --user --name eb --display-name "Python (eb)"
R -q -e "IRkernel::installspec(name = 'ir-eb', displayname = 'R (eb)')"
