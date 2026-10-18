#!/usr/bin/env bash
# The AutoEval regression gate (ADR 0013 D4): the pinned judge on the first published VideoPhy-2
# test videos against the published scores. Run setup_autoeval.sh first; exit 1 when it fails.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$REPO_ROOT"
source_h200_env

require_cuda
export PHYSVERIFY__H3__AUTOEVAL__ROOT="${PHYSVERIFY__H3__AUTOEVAL__ROOT:-$H200_ROOT}"
pv h3 gate-autoeval
