# Template of $PHYSVERIFY_H200_ROOT/env.sh, the machine-local environment of the H200 (not in git):
#   cp scripts/h200/env.example.sh "$PHYSVERIFY_H200_ROOT/env.sh"
# run_kubric.sh, run_demo.sh, run_generator_comparison.sh, setup_autoeval.sh and smoke_autoeval.sh source it
# (source_h200_env in _common.sh); source it by hand before the other scripts. Every path derives
# from the directory the file sits in, so the copy needs no editing.
H200_ENV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PHYSVERIFY_H200_ROOT="$H200_ENV_ROOT"
export HOME="$H200_ENV_ROOT"
export XDG_CACHE_HOME="$H200_ENV_ROOT/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export UV_PYTHON_INSTALL_DIR="$H200_ENV_ROOT/.local/python"
export PATH="$H200_ENV_ROOT/.local/bin:$PATH"
# The host Python's Datadog ddtrace breaks udocker and slows every Python start.
export DD_TRACE_ENABLED=false
unset H200_ENV_ROOT
