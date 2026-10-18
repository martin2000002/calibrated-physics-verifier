#!/usr/bin/env bash
# Preference phase (ADR 0012): prompts, base candidates of train and dev, scores, pairs.
# Every step resumes, so the script can simply be run again after an interruption.
# PHYSVERIFY_FINETUNE_PILOT=1 runs the pilot only: 20 train prompts x K = 4, scored and paired.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$REPO_ROOT"

PILOT_CANDIDATES=80

require_cuda
bash scripts/h200/setup.sh
pv finetune prompts build
if [[ "${PHYSVERIFY_FINETUNE_PILOT:-0}" == "1" ]]; then
    pv finetune generate --split train --limit "$PILOT_CANDIDATES"
    pv finetune score --split train
else
    pv finetune generate --split train
    pv finetune generate --split dev
    pv finetune score --split train
    pv finetune score --split dev
fi
pv finetune pairs
pv finetune status
