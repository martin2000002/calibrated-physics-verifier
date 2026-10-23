# H200 scripts

Runners for the university H200 server. **Generation and every cost measurement run only on the
H200; runs on any other machine are harness checks, not measurements.** A run counts as a
measurement only with `--device cuda` and the `precise` profile. `pv` in `_common.sh` passes both
by default, and `run_timing.sh`, `run_preference.sh`, `run_finetune.sh`,
`run_generator_comparison.sh` and `smoke_autoeval.sh` keep them. The other runners do not:
`smoke_finetune.sh` uses the `fast` profile; `run_freeze.sh`, `run_final.sh` and
`run_human_agreement.sh` run on the CPU; `run_perception_iteration.sh` takes its device from
`PHYSVERIFY_ITERATION_DEVICE` (default `cpu`) and its profile from `PHYSVERIFY_ITERATION_PROFILE`;
`run_kubric.sh` runs on the CPU with the profile of `PHYSVERIFY_KUBRIC_PROFILE`; and `run_demo.sh`
scores a given video with `--device cpu` when there is no GPU.

Each script's header lists its commands or steps, its startup checks (every refusal exits 2
before any `physverify` call) and its variables with their defaults; this file gives the order of
use and what each script leaves behind.

| Script | Purpose |
|---|---|
| `setup_cuda.sh` | CUDA environment, perception weights and the Wan2.1 snapshot |
| `run_timing.sh` | Timing benchmark |
| `smoke_finetune.sh`, `run_preference.sh`, `run_finetune.sh` | Preference pairs and Flow-DPO fine-tuning |
| `run_kubric.sh` | Kubric synthetic dataset (Docker or udocker) |
| `run_perception_iteration.sh`, `perception_arms.tsv` | One iteration of the perception campaign on the calib clips |
| `run_freeze.sh`, `freeze_query.py` | The freeze: adopted iteration, exclusion record, abstention level, `freeze.json` |
| `run_final.sh` | The sealed final looks |
| `run_human_agreement.sh`, `agreement_query.py` | Human-agreement study on VideoPhy (CPU) |
| `setup_autoeval.sh`, `smoke_autoeval.sh`, `autoeval_requirements.txt` | VideoPhy-2 AutoEval judge and its gate |
| `run_generator_comparison.sh` | Fine-tuned vs base vs best-of-8 |
| `run_demo.sh` | End-to-end demo |
| `run_query.py` | Read-only queries the scripts ask of the checkout (`physverify.app.queries`) |
| `_common.sh`, `_steps.sh`, `_guards.sh`, `env.example.sh` | Shared helpers (sourced) and the environment template |

## Machine-local root and environment

Every H200 path derives from one root, `PHYSVERIFY_H200_ROOT` (default `/home/mmonterot/work`,
set in `_common.sh`): the environment file `$PHYSVERIFY_H200_ROOT/env.sh`
(`PHYSVERIFY_H200_ENV` overrides it), the Kubric data directory, the Kubric lock, the udocker
binary and the output directories below. The environment file is not in git; its template is
`env.example.sh`:

```sh
cp scripts/h200/env.example.sh "$PHYSVERIFY_H200_ROOT/env.sh"
source "$PHYSVERIFY_H200_ROOT/env.sh"   # HOME, XDG_CACHE_HOME, HF_HOME, UV_*, PATH, DD_TRACE_ENABLED
```

It derives every path from the directory it sits in. `run_kubric.sh`, `run_demo.sh`,
`run_generator_comparison.sh`, `setup_autoeval.sh`, `smoke_autoeval.sh`,
`run_perception_iteration.sh`, `run_freeze.sh`, `run_final.sh` and `run_human_agreement.sh` source
it when present (`source_h200_env` in `_common.sh`); source it by hand before the other scripts
(`run_timing.sh`, `setup_cuda.sh`, `smoke_finetune.sh`, `run_preference.sh`, `run_finetune.sh`).
The first runs used the root `/tmp/mmonterot`, which the 30-day purge of `/tmp` can wipe;
`PHYSVERIFY_H200_ROOT=/tmp/mmonterot` still reproduces that layout.

## Timing

Requirements: a CUDA 12.8 capable NVIDIA driver (`nvidia-smi -L` must list the GPU), `uv` on
`PATH`, and disk for the Hugging Face cache (`paths.hf_home`): about 29 GB for the pinned
Wan2.1-T2V-1.3B snapshot plus the perception weights. Docker is optional: when `docker info`
responds, the Kubric image is pulled if missing and the `smoke` scenes are generated, simulated and
perturbed so `kubric_render` can be timed; otherwise that target is skipped with its reason.

```sh
bash scripts/h200/run_timing.sh
```

It runs, from the repository root:

1. `setup_cuda.sh`: `uv sync --no-default-groups --group dev --group cu128 --group perception
   --group generation --group data`, a check that torch sees CUDA, `extract fetch-weights` for the
   `precise` and the `fast` profile, and `timing fetch` (the pinned Wan2.1 snapshot; a cut
   download resumes).
2. `physverify timing env`: the environment of the run and whether it is a valid measurement.
3. The Kubric prerequisites (only when Docker responds).
4. One `physverify timing run --dataset local_video --subset timing_wan21` with every target of
   `timing.order`. `wan21_generate` writes the generated videos and their `manifest.json` to
   `data/raw/local_video/timing_wan21/`; `extract_pipeline` and `verify_checks` then time those same
   clips.

Every `physverify` call gets `--profile precise --device cuda --set runtime.threads=N`, with `N`
the CPUs available to the job (`nproc`); `PHYSVERIFY_TIMING_THREADS` overrides it. The script exits
with the exit code of `timing run`, and with 2 before calling `uv` when there is no CUDA GPU.

Results: `data/runs/<run_id>/timing/` (`results.json`, `samples.csv`, `stats.csv`, `summary.md`),
the console log `data/runs/<run_id>.log` and the link `data/runs/h200-timing-latest`. An interrupted
run keeps the samples written so far; `timing summarize` rebuilds the statistics, and `timing
budget` turns a valid run into the number of training prompts and pairs that fit a GPU budget:

```sh
uv run --no-sync physverify timing summarize --run data/runs/h200-timing-latest
uv run --no-sync physverify timing budget --run data/runs/h200-timing-latest \
    --gpu-hours <H> --candidates <K> --eval-prompts <Q>
```

## Fine-tuning

Preference pairs from the verifier and a Flow-DPO LoRA of Wan2.1-T2V-1.3B. Every script needs a
CUDA GPU (exit 2 otherwise) and resumes after an interruption: run it again.

```sh
bash scripts/h200/smoke_finetune.sh                              # fast profile, end to end
PHYSVERIFY_FINETUNE_PILOT=1 bash scripts/h200/run_preference.sh   # pilot: 20 train prompts x K = 4
bash scripts/h200/run_preference.sh                              # all train and dev candidates
bash scripts/h200/run_finetune.sh                                # beta grid, training, selection
```

1. `smoke_finetune.sh`: the `fast` profile on the campaign `wan21_smoke`, with
   `finetune.prompts.n_train=8` and `val_fraction=0.5` (two prompts per family, so the per-family
   val split keeps one of each), `finetune.candidates.k_train=4`, the provisional scorer and
   relaxed pair rules (`min_margin`, `min_motion`, `sigma_k` and `min_motion_ratio` at 0, no
   object-count requirement, `max_pairs_per_prompt=2`); then 5 training steps, a check that the
   val `kl_proxy` moved off 0, and one video generated with the trained adapter.
2. `run_preference.sh`: `finetune prompts build`, `generate` and `score` of the train (and dev)
   split, then `finetune pairs`. After the pilot, record the hand review of its pairs with
   `physverify finetune pilot-review --sample` (prints the pairs to inspect) and then `--better
   N`. The stop rule (`[finetune.stop_rule]` in `configs/studies/finetune.toml`,
   `finetune/stop_rule.py`) fires when pairs per scored prompt are below 0.3, abstention per
   candidate is above 50 %, no review of these pairs is recorded, or fewer than 6 of the 10
   inspected pairs show the better video as winner. `finetune train`, `finetune beta-grid`,
   `compare-generators generate` and `compare-generators freeze` then exit 2 unless
   `--stop-rule-deviation TEXT` is given or `finetune.stop_rule.enforce` is false; the verdict
   goes to `pairs/<scorer>/stop_rule.json` either way.
3. `run_finetune.sh`: `finetune beta-grid` (4 short runs of a quarter of the main run each, chosen
   on val pairs), `finetune train` (warmup and checkpoint spacing are fractions of the run's
   steps; it exits 2 when the schedule would leave fewer checkpoints than `select` compares) and
   `finetune select` (dev scored for spaced checkpoints against the base).

`train` and `beta-grid` refuse pairs from the provisional scorer unless `--allow-provisional` is
passed, and the calibrated scorer is unavailable (exit 2) until the calibration artifact has status
`calibrated`. `train --no-resume` refuses a run id that holds checkpoints unless `--force` is also
given, which deletes them.

Results live under `data/finetune/<campaign>/`: `prompts.jsonl`, `candidates/<model_tag>/`,
`scores/`, `pairs/<scorer>/` (`pairs.jsonl`, `val_pairs.jsonl`, `report.json`),
`train/<run_id>/` (`step-<n>/`, `metrics.csv`, `val.csv`, `run.json`) and `select/<run_id>/`.
`physverify finetune status` summarises a campaign, and `physverify finetune budget --timing-run
<dir> --gpu-hours <H>` turns a valid timing run with the `dpo_step` target into the number of
training prompts that fit.

## Kubric dataset

The Kubric tasks run on CPU in the pinned image (`runtime.docker_image`), through Docker by
default or through [udocker](https://github.com/indigo-dc/udocker) in a named container created
from that same pinned digest when Docker is not available (`PHYSVERIFY_KUBRIC_ENGINE=udocker`; the
`[container]` table, see `configs/local.example.toml`). No GPU is needed.

```sh
# smoke: fast profile, 2 clips, fresh data directory
PHYSVERIFY_KUBRIC_PROFILE=fast PHYSVERIFY__PATHS__DATA_DIR="$PHYSVERIFY_H200_ROOT/kubric-smoke" \
    bash scripts/h200/run_kubric.sh smoke
# full dataset, precise profile, in the background
nohup bash scripts/h200/run_kubric.sh full >/dev/null 2>&1 &
```

`run_kubric.sh [subset]` (default `full`) sources the environment file when present and exports:

- the four `DD_*=false` variables, without which the host Python's ddtrace breaks udocker;
- `PHYSVERIFY__CONTAINER__ENGINE` from `PHYSVERIFY_KUBRIC_ENGINE` (`docker`, the default, or
  `udocker`), plus `..._UDOCKER_BIN` (`PHYSVERIFY_UDOCKER`, default
  `$PHYSVERIFY_H200_ROOT/udocker-venv/bin/udocker`) and `..._UDOCKER_CONTAINER`
  (`PHYSVERIFY_UDOCKER_CONTAINER`, default `kub`) for udocker;
- `PHYSVERIFY__PATHS__DATA_DIR` (default `$PHYSVERIFY_H200_ROOT/kubric-data`, outside the
  repository root so the laptop's `rsync --delete` cannot touch it);
- a Kubric-only heavy lock (`PHYSVERIFY_KUBRIC_LOCK`, default `$PHYSVERIFY_H200_ROOT/kubric.lock`,
  timeout 0): a second build on the same lock exits 75.

Preflight (exit 2 on failure): the data directory has at least `PHYSVERIFY_KUBRIC_MIN_FREE_GB`
(50) GB free; `physverify doctor` reports the chosen `container_engine`; for Docker, `docker info`
responds and the pinned image is pulled with `physverify synth pull-image` when missing; for
udocker, the binary exists and the container exists (otherwise the image is pulled and the
container created). The udocker image is `runtime.docker_image` unless `PHYSVERIFY_KUBRIC_IMAGE`
is set, and the image `udocker ps` lists for the container must equal it: a container from another
image, or one whose image cannot be read, stops the script; remove it with `udocker rm kub` so the
next run recreates it from the pin, or set `PHYSVERIFY_KUBRIC_IMAGE` to its image to use it
knowingly. Last, `physverify doctor --probe-container` runs the configured engine with the build's
own mounts, workdir and environment and imports `kubric`, `bpy`, `pybullet` and
`physverify.kubric_worker.build_scene` from the mounted source.

Then it runs `physverify synth build` and `synth list` with `--set
runtime.threads=$PHYSVERIFY_KUBRIC_THREADS` (16, also Blender's fixed thread count),
`dataset.render.workers=$PHYSVERIFY_KUBRIC_WORKERS` (12) and
`dataset.render.jobs_per_process=$PHYSVERIFY_KUBRIC_JOBS` (8); render settings do not enter the
dataset hash. The log goes to `<data_dir>/runs/kubric-<subset>-<UTC time>.log` (`kubric-latest.log`
links to it) and starts with the engine and the image the container came from; `synth list`
always runs, and the script exits with the build's exit code. When a run fails or is cut, run the
same command again: every step skips finished items. If udocker failed and Docker responds,
rerun it with `PHYSVERIFY_KUBRIC_ENGINE=docker`, keeping the same data directory.

### Ground-truth migrations

`physverify synth migrate --name <name>` re-runs APPLY on every simulated clip of a stored dataset
in place (no version bump), keeping every render whose inputs did not change, and writes
`migrations/<name>.json`. `velocity-position` (the default, also `synth migrate-velocity`) moves a
dataset to `velocity_convention = "position"`; the ground-truth fixes of `plan.GT_MIGRATIONS`
follow, in order: `floor-crossing` re-solves the floor bounces whose row after the crossing was
already under the floor, and `pair-collision` holds a ball's velocity at a ball-ball collision only
when a floor record overlaps the row's interval or the ball is within `contacts.tol_m` of the floor
(both need a dataset on `velocity-position`). A fix runs only as the first one the plan lacks and
covers every later one, since APPLY re-runs with the checkout's code. Afterwards rebuild every
listed subset (only the renders whose poses changed are redone) and `extract purge` the
re-rendered clips. A plan whose `gt_revision` is older than the checkout's is refused by every
build step until its pending fix (named in the refusal) has run on it.

## Perception campaign iterations

```sh
source "$PHYSVERIFY_H200_ROOT/env.sh"
PHYSVERIFY_ITERATION_ID=it0-baseline PHYSVERIFY_ITERATION_EXPECT_COMMIT=<head> \
    PHYSVERIFY_ITERATION_CACHE_DIR=<shared cache root> \
    nohup bash scripts/h200/run_perception_iteration.sh > "$PHYSVERIFY_H200_ROOT/iterations/it0.log" 2>&1 &
```

`run_perception_iteration.sh [FROM [TO]]` runs five steps on the `kubric_synthetic` calib clips
only: `extract run`, `diagnose stages --write-profile` (the iteration's own error profile),
`extract run` again (lifting reads the new profile), `calibrate stage-sigma` and `calibrate
dev-score --level both`, the development metric (the monotonicity of the full calib fit and the
informedness of out-of-fold decisions per check, the out-of-fold AUROC of the clip score).
Everything lands in `$PHYSVERIFY_H200_ROOT/iterations/<id>-<device>/` (calibration directory,
runs, heavy lock, `devscore.json`, the full calib fit `calibration.json`, `summary.md`).
`PHYSVERIFY_ITERATION_PROFILE` (default `precise`) picks the sampling profile, which also fixes the
dataset version the calib clips are listed from. With `PHYSVERIFY_ITERATION_DEVICE=cuda`, launch it
with one GPU, `CUDA_VISIBLE_DEVICES` in 0-3 (anything else, a list included, exits 2), through
`gpu_queue.sh`, the server's GPU queue launcher (not in this repository).

Declared arms: `PHYSVERIFY_ITERATION_ARM=<arm>` names a line of `perception_arms.tsv`, whose
provider swaps (`extraction.plan.<stage>=<provider>` only) go as `--set` to every physverify call.
The profile stays the committed one, so arms that share a provider prefix share its cache keys and
its error-profile entries (a separate profile name would not: the name is part of every entry).
`doctor` must list each swap among its sources, `state.env` records `CHAIN_ARM` (a resume with
another arm exits 2) and the summary names the arm and the GPU. Give each arm its own
`PHYSVERIFY_ITERATION_ID` containing the arm name (`it<n>-<arm>`), and run arms one at a time.

### Sealed sets

The sealed ledger (`configs/verifier/sealed.toml`): `calibrate fit|stage-sigma|evaluate|dev-score`,
`selective-risk fit|curve`, `verify`, `diagnose *` and `scene-stats per-clip` exit 2 before
computing on a sealed set (`kubric_synthetic:eval`, `kubric_fresh`, `kubric_fresh_ood`,
`kubric_fresh_blind`, `physics_iq_views`, `objectron_static`) unless `PHYSVERIFY_UNSEAL`
names it, and then append a line to the ledger first; the read and the append hold one flock on
the sibling `<ledger>.lock`. A look never creates the ledger: create it
once, by hand, with `env.sh` sourced (`touch "$PHYSVERIFY_H200_ROOT/sealed-ledger.txt"`); a missing
file (a wrong `HOME`, another path) stops the look, and stderr prints the path it resolved. A look
needs a clean git checkout (nothing uncommitted under `src` or `configs`). Its frozen state is the
commit, the configuration sections with profile and device (not paths, threads or the lock) and
the sha256 of `calibration.json` (or the `--artifact` used), `thresholds.json`, `exclusions.json`
and the error profile; a look under another frozen state, a refit at the same commit included,
also needs `PHYSVERIFY_UNSEAL_REPEAT=1`. A missing `[sealed]` table fails closed.

## Freeze and final looks

Both run in a clean checkout at the freeze commit, on the CPU, profile `precise`, with the
committed configs only.

```sh
source "$PHYSVERIFY_H200_ROOT/env.sh"
PHYSVERIFY_FREEZE_EXPECT_COMMIT=$C PHYSVERIFY_FREEZE_ITERATION=<chosen iteration id> \
    PHYSVERIFY_FREEZE_ITERATIONS="<iteration dirs>" \
    PHYSVERIFY_FREEZE_EARLY_EXTRACTION_COMMIT=<early-extraction commit> \
    PHYSVERIFY_FREEZE_V1_COMMIT=<v1 commit> PHYSVERIFY_FREEZE_V1_CALIBRATION=<base calibration.json> \
    PHYSVERIFY_FREEZE_CACHE_DIR=<cache> \
    nohup bash scripts/h200/run_freeze.sh > "$PHYSVERIFY_H200_ROOT/final/freeze.log" 2>&1 &
```

`run_freeze.sh [FROM [TO]]` runs seven steps: it adopts the iteration the selection rule chooses
(`freeze_query.py select`; never refitted), extracts the development real video, evaluates calib
with the real datasets, verifies the R2 clips with both detector arms, runs `exclusions decide`
and `selective-risk fit`, and writes `$F/freeze.json` (`freeze_query.py manifest`: commit, configs
tree, artifact sha256, `record_sha`, the seed of the fresh renders, the two `freeze_query.py diff`
results and the v1 arm's commit and calibration). `$F` defaults to `$PHYSVERIFY_H200_ROOT/final`.

```sh
export PHYSVERIFY_FINAL_EXPECT_COMMIT=$C PHYSVERIFY_FINAL_SEED=<preregistered seed>
bash scripts/h200/run_final.sh seed
bash scripts/h200/run_final.sh render kubric_fresh          # then kubric_fresh_ood
bash scripts/h200/run_final.sh fetch physics_iq_views       # and objectron_static
bash scripts/h200/run_final.sh extract kubric_fresh         # each set, real_new as its two sets
bash scripts/h200/run_final.sh look kubric_fresh            # once per set; real_new together
bash scripts/h200/run_final.sh v1arm kubric_fresh           # kubric_synthetic_eval, real_new
```

`run_final.sh` checks the freeze (commit, device, profile, calibration directory, frozen files,
seed) on every command. It refuses a set whose cache existed before its first extraction, a Kubric
set without the render of the seed, a render after a look, a second look at any set the ledger
covers (`look SET --retry` passes once, when that look wrote no evaluation) and changed downloads.
Bookkeeping, logs and per-command summaries land in `$F/run-final/`; commands on different sets may
run at the same time (each holds its own locks; the heavy lock is `$F/heavy.lock`). The v1 arm runs
at the v1 commit through `overlay_cli.py` (`PHYSVERIFY_FINAL_OVERLAY_CLI`; a server-side script,
not in this repository) with the freeze's configs and the base `checks.toml`, the base calibration
and `PHYSVERIFY_UNSEAL_REPEAT=1`, and must leave no `meta.json` in its overlay. `PHYSVERIFY_FINAL_DRYRUN=1` is the rehearsal: `$F`, the data,
the cache and its own ledger (`PHYSVERIFY__SEALED__LEDGER`) under `$PHYSVERIFY_H200_ROOT/scratch`,
a seed other than the freeze's, never `kubric_synthetic_eval`, and, for the new real control,
`PHYSVERIFY__PHYSICS_IQ_VIEWS__VIEWS='["center"]'` and `PHYSVERIFY__OBJECTRON_STATIC__SPLIT=train`.
A run with `PHYSVERIFY_FINAL_EXTRACT_LIMIT` (the pace measurement) is recorded as
`extract-limited`: a look still needs a full `extract`.

Look order (fixed in the study plan; the script enforces only that a v1 arm follows its look and
that no render follows a look, so this order is kept by hand): `look kubric_synthetic_eval` (the
set already seen), `look kubric_fresh`, `run_human_agreement.sh dev` then `test`, `look real_new`;
afterwards, with the same freeze and seed, the `kubric_fresh_ood` render and look, `look
kubric_fresh_blind` and the v1 arms.

## Human agreement

```sh
source "$PHYSVERIFY_H200_ROOT/env.sh"
export PHYSVERIFY_AGREEMENT_EXPECT_COMMIT=$C \
    PHYSVERIFY_AGREEMENT_FREEZE=$F/freeze.json PHYSVERIFY_AGREEMENT_CALIBRATION_DIR=$F/calibration
bash scripts/h200/run_human_agreement.sh dev    # read summary.md before going on
bash scripts/h200/run_human_agreement.sh test   # runs once
```

Two phases a person runs in turn, on the CPU (`--profile precise --device cpu`, the cache keys of
the VideoPhy clips) through `$PHYSVERIFY_AGREEMENT_PYTHON`, with no `uv` and no GPU. `dev` fetches,
extracts and verifies the dev subset and freezes the rule, gate and cut into the pre-registration;
`test` does the same on the test subset and runs the one pre-registered `agreement compare --rule
calibrated`. With `PHYSVERIFY_AGREEMENT_FREEZE` it reads the freeze manifest (`freeze.json` of
`run_freeze.sh`); without it, the CPU calibration chain in `PHYSVERIFY_AGREEMENT_CHAIN_DIR`, the
stored output directory of a calibration-chain runner that is no longer in the repository
(default: `CHAIN_ROOT` of `_common.sh`, under `$PHYSVERIFY_H200_ROOT`). `test` refuses once any
comparison is logged, also through the list of test logs in
`$PHYSVERIFY_H200_ROOT/h2-test-ledger.txt`, and `dev` refuses on the same evidence. A pre-registration with a calibration gate makes the study not evaluable with the
calibrated rule; `test` then needs `PHYSVERIFY_AGREEMENT_ACCEPT_NOT_EVALUABLE=1`. Output goes to
`PHYSVERIFY_AGREEMENT_DIR` (default `$PHYSVERIFY_H200_ROOT/h2`). Unlike the freeze, iteration and
final runners, this script keeps its own startup checks instead of `_guards.sh`.

## Generator comparison

```sh
source "$PHYSVERIFY_H200_ROOT/env.sh"
bash scripts/h200/run_generator_comparison.sh base-eval   # base eval videos; outcome-blind, can start first
bash scripts/h200/setup_autoeval.sh                       # VideoPhy-2 AutoEval in its own venv (no GPU)
bash scripts/h200/smoke_autoeval.sh                       # the gate: exit 1 when the pinned judge fails it
bash scripts/h200/run_generator_comparison.sh dev         # AutoEval on dev videos (after `finetune select`)
bash scripts/h200/run_generator_comparison.sh freeze      # pre-registration, before any eval score
bash scripts/h200/run_generator_comparison.sh eval        # fine-tuned videos, scores, eval judge rows, compare
bash scripts/h200/run_generator_comparison.sh status
```

The order is fixed: `freeze` writes `h3/preregistration.json`, and it must exist before any eval
score or judge row; `score`, `judge --split eval` and `compare` refuse to run without it. When
`compare-generators status` does not show `gate: passed`, `eval` skips the judge and still runs
`compare` (verdict at most `supported_verifier_only`); any other failure stops the script. The eval
split belongs to `compare-generators`: `finetune generate|score --split eval` exit 2. `base-eval`
and `dev` pass extra arguments to the `physverify compare-generators` subcommand (`--shard I/N`,
`--limit N`, `--task pc|sa`). Every step skips the work it finds done with the same stamp.

`setup_autoeval.sh` is idempotent. It reads the pins from `[h3.autoeval]` in
`configs/studies/generator_comparison.toml` (upstream `code_commit`, `hf_repo` at `hf_revision`),
creates a Python 3.10 venv, checks the upstream clone out at that commit and installs
`autoeval_requirements.txt`. It downloads the model, writes `checkpoint.sha256` and stores `uv pip
freeze` in `freeze_file`, whose sha256 goes into the gate report and the judge stamp. The relative
paths sit under `PHYSVERIFY__H3__AUTOEVAL__ROOT` (default `$PHYSVERIFY_H200_ROOT`). Only
`transformers==4.28.1` is an upstream pin; the torch, peft, numpy and Python versions are our
choice, and the freeze file records the rest. `PHYSVERIFY_AUTOEVAL_UPSTREAM` overrides the clone
URL. The GPU steps exit 2 without a CUDA GPU before any `uv` call.

## Demo

```sh
bash scripts/h200/run_demo.sh scenes      # after `compare-generators freeze` and the eval generation
bash scripts/h200/run_demo.sh run --prompt "A ball rolls off a table, static camera" \
    --query ball:1 --query table:1        # generates k = 0 (needs CUDA), then verifies it
bash scripts/h200/run_demo.sh run --prompt "..." --query ball --video clip.mp4   # no GPU needed
```

With `--video` and no GPU (`nvidia-smi -L` lists none) the script scores the given video with
`--device cpu`, which is slow but valid; with a GPU it keeps `--device cuda`. `scenes` and the
generating `run` exit 2 without a GPU.

Each video gets `<out>/<prompt_id>__<model_tag>/{report.json,report.md,overlay.mp4}`; `scenes`
adds `<prompt_id>/side_by_side.mp4` and `scenes.md`. `<out>` is `PHYSVERIFY_DEMO_OUT` (default
`$PHYSVERIFY_H200_ROOT/demo`). Ad-hoc prompts and given videos go to the `demo_adhoc` campaign,
never to the comparison's; `scenes` exits 2 until `h3/preregistration.json` exists or while a k = 0
video is missing. `scenes` scores with `finetune.scorer` only, never the provisional fallback, and
exits 2 when that scorer is unavailable or its label is not the `rule_label` frozen in the
pre-registration (then refreeze). `run` and `report` fall back to the provisional rule, labelled as
such, while `demo.allow_provisional` is on. `physverify demo report --results <dir>` rebuilds a
report on CPU, for example once the calibrated rule replaces the provisional one.
