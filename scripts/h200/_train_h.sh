# The identity of subset train_h (version and seed label), shared by its build (run_train_h.sh)
# and the training of the learned variants on it (run_learned_h.sh); sourced, never executed.

TRAIN_H_SEED_LABEL="plan-humano train v1"
TRAIN_H_VERSION="v2-train-h"

# Exports the dataset settings that make kubric_synthetic/train_h this set: its own version, the
# seed derived from the label and every scene split=train.
export_train_h_dataset() {
    local seed
    seed="$(PYTHONPATH="$REPO_ROOT/src" uv run --no-sync python -c \
        "import sys; from physverify.synth.seeds import labelled_seed; print(labelled_seed(sys.argv[1]))" \
        "$TRAIN_H_SEED_LABEL")" || die "could not derive the dataset seed"
    export PHYSVERIFY__DATASET__VERSION="$TRAIN_H_VERSION"
    export PHYSVERIFY__DATASET__SEED="$seed"
    export PHYSVERIFY__DATASET__CALIB_FRACTION=0.0
    export PHYSVERIFY__DATASET__TRAIN_SPLIT=true
}
