#!/usr/bin/env bash
# Installs the CUDA environment and downloads every weight the H200 benchmark needs.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$REPO_ROOT"

require_cuda
uv sync --no-default-groups --group dev --group cu128 --group perception --group generation \
    --group data
uv run --no-sync python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 2)" \
    || die "torch cannot see CUDA; check the driver (CUDA 12.8) and the cu128 group"
pv extract fetch-weights
# The fine-tuning smoke scores with the fast profile, whose providers differ from precise.
uv run --no-sync physverify --profile fast extract fetch-weights
pv timing fetch
