#!/usr/bin/env bash
set -euo pipefail

conda run -n conformal_training python3 --version

CONDA_BASE="${HOME}/miniconda3"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate conformal_training

python3 --version
