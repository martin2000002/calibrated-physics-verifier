# Startup guards of the H200 runners that must run the committed configuration of a clean
# checkout (run_iteration.sh, run_freeze.sh, run_final.sh); sourced after _common.sh, never
# executed. Each guard exits 2 (die) before any physverify call.

# near PATH BASE: PATH is a directory named BASE* (BASE, BASE-cuda, ...) or below one.
near() {
    [[ "$(realpath -m "$1")" == "$(realpath -m "$2")"* ]]
}

# under PATH DIR: PATH is strictly below DIR.
under() {
    [[ "$(realpath -m "$1")" == "$(realpath -m "$2")"/* ]]
}

# refuse_foreign_config SCRIPT OWN [ALLOWED]: no PHYSVERIFY__* variable but the names of OWN
# (without the prefix, space separated, with a leading and a trailing space) or those matching
# the regular expression ALLOWED.
refuse_foreign_config() {
    local script="$1" own="$2" allowed="${3:-}" foreign="" name
    for name in $(compgen -e -X '!PHYSVERIFY__*'); do
        [[ "$own" == *[[:space:]]"${name#PHYSVERIFY__}"[[:space:]]* ]] && continue
        [[ -n "$allowed" && "$name" =~ $allowed ]] && continue
        foreign+=" $name"
    done
    [[ -z "$foreign" ]] || die "the environment sets configuration $script does not own:" \
        "$foreign (the run would not be the committed one); unset it"
}

# refuse_local_config SCRIPT: only $REPO_ROOT/configs, and no local.toml in it.
refuse_local_config() {
    [[ -z "${PHYSVERIFY_CONFIG_DIR+set}" ]] \
        || die "PHYSVERIFY_CONFIG_DIR is set: $1 reads only $REPO_ROOT/configs; unset it"
    [[ ! -e "$REPO_ROOT/configs/local.toml" ]] \
        || die "$REPO_ROOT/configs/local.toml exists: it is not committed; remove it"
}

# require_checkout VAR: HEAD starts with the value of VAR (>= 7 characters) and nothing is
# uncommitted or untracked under src, configs, scripts, annotations, pyproject.toml or uv.lock.
# Sets HEAD_COMMIT.
require_checkout() {
    local expect="${!1:-}" dirty
    [[ ${#expect} -ge 7 ]] || die "set $1 to the commit this run uses"
    HEAD_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)" || die "$REPO_ROOT is not a git checkout"
    [[ "$HEAD_COMMIT" == "$expect"* ]] || die "HEAD is $HEAD_COMMIT, expected $expect"
    dirty="$(git -C "$REPO_ROOT" status --porcelain -- src configs scripts annotations pyproject.toml uv.lock)" \
        || die "git status failed in $REPO_ROOT"
    [[ -z "$dirty" ]] || die "uncommitted or untracked files under src, configs, scripts," \
        "annotations, pyproject.toml or uv.lock:"$'\n'"$dirty"
}

# check_frozen_doctor DIR CAL CACHE: doctor (through pv) reports the committed configs, device cpu,
# profile precise, paths.cache_dir CACHE, the error profile in CAL, and the calibration directory,
# runs directory and thresholds inside DIR. Sets DOCTOR.
check_frozen_doctor() {
    local name
    DOCTOR="$(pv doctor)" || die "physverify doctor failed"
    same_path "$(doctor_value config_dir)" "$REPO_ROOT/configs" \
        || die "doctor reports config_dir '$(doctor_value config_dir)', not $REPO_ROOT/configs"
    [[ "$(doctor_value device)" == cpu* ]] || die "doctor reports device '$(doctor_value device)', not cpu"
    [[ "$(doctor_value profile)" == precise ]] \
        || die "doctor reports profile '$(doctor_value profile)', not precise"
    same_path "$(doctor_value paths.cache_dir)" "$3" \
        || die "doctor reports paths.cache_dir '$(doctor_value paths.cache_dir)', not $3"
    for name in calibration_dir runs_dir error_profile thresholds; do
        under "$(doctor_value "paths.$name")" "$1" \
            || die "doctor reports paths.$name '$(doctor_value "paths.$name")', outside $1"
    done
    same_path "$(doctor_value paths.error_profile)" "$2/stage_errors.precise.json" \
        || die "doctor reports paths.error_profile '$(doctor_value paths.error_profile)', not" \
            "$2/stage_errors.precise.json"
}
