#!/usr/bin/env bash
# VideoPhy-2 AutoEval in its own environment (ADR 0013 D4); idempotent, needs no GPU.
# Pins come from [h3.autoeval] in configs/studies/generator_comparison.toml; relative paths are
# under PHYSVERIFY__H3__AUTOEVAL__ROOT (default $PHYSVERIFY_H200_ROOT). Steps: a Python 3.10 venv,
# the upstream clone checked out at code_commit, autoeval_requirements.txt, the model at
# hf_revision, a sha256 manifest of the model files and `uv pip freeze` into freeze_file.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
cd "$REPO_ROOT"
source_h200_env

UPSTREAM="${PHYSVERIFY_AUTOEVAL_UPSTREAM:-https://github.com/Hritikbansal/videophy.git}"
AUTOEVAL_ROOT="${PHYSVERIFY__H3__AUTOEVAL__ROOT:-$H200_ROOT}"
PINS="configs/studies/generator_comparison.toml"

# The string value of KEY in [h3.autoeval] of $PINS.
autoeval_value() {
    local value
    value="$(sed -n '/^\[h3\.autoeval\]/,/^\[/{s/^'"$1"' *= *"\([^"]*\)".*/\1/p}' "$PINS")"
    [[ -n "$value" ]] || die "$PINS: [h3.autoeval] $1 is not set"
    printf '%s\n' "$value"
}

under_root() {
    [[ "$1" == /* ]] && printf '%s\n' "$1" || printf '%s\n' "$AUTOEVAL_ROOT/$1"
}

VENV="$(under_root "$(autoeval_value venv)")"
CLONE="$(under_root "$(autoeval_value repo_dir)")"
CKPT="$(under_root "$(autoeval_value checkpoint_dir)")"
FREEZE="$(under_root "$(autoeval_value freeze_file)")"
CODE_COMMIT="$(autoeval_value code_commit)"
HF_REPO="$(autoeval_value hf_repo)"
HF_REVISION="$(autoeval_value hf_revision)"
PYTHON="$VENV/bin/python"

command -v uv >/dev/null 2>&1 || die "uv not found; source \$PHYSVERIFY_H200_ROOT/env.sh"
command -v git >/dev/null 2>&1 || die "git not found"

[[ -x "$PYTHON" ]] || uv venv --python 3.10 "$VENV"

if [[ ! -d "$CLONE/.git" ]]; then
    if [[ -e "$CLONE" ]] && [[ -n "$(ls -A "$CLONE")" ]]; then
        die "$CLONE exists and is not an empty directory or a clone"
    fi
    mkdir -p "$(dirname "$CLONE")"
    git clone --quiet "$UPSTREAM" "$CLONE"
fi
git -C "$CLONE" checkout --quiet --detach "$CODE_COMMIT"
[[ "$(git -C "$CLONE" rev-parse HEAD)" == "$CODE_COMMIT" ]] \
    || die "$CLONE is not at $CODE_COMMIT"

uv pip install --python "$PYTHON" -r scripts/h200/autoeval_requirements.txt

"$PYTHON" -s -E -c '
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])
' "$HF_REPO" "$HF_REVISION" "$CKPT"

mkdir -p "$(dirname "$FREEZE")"
(cd "$CKPT" && find . -type f ! -path './.cache/*' | LC_ALL=C sort | xargs -d '\n' sha256sum) \
    > "$(dirname "$FREEZE")/checkpoint.sha256"
uv pip freeze --python "$PYTHON" > "$FREEZE"
echo "autoeval: $CLONE at $CODE_COMMIT, $HF_REPO at $HF_REVISION in $CKPT"
echo "freeze: $FREEZE"
