#!/usr/bin/env bash
# Fine-tuning phase (ADR 0012): beta grid on val pairs, Flow-DPO LoRA, checkpoint selection on dev.
# Every step resumes from its last checkpoint or its stored scores.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$REPO_ROOT"

require_cuda
pv finetune status
pv finetune beta-grid
pv finetune train --resume
pv finetune select
pv finetune status
