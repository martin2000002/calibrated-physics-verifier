# H200 timing benchmark

**Generation and every cost measurement run only on the H200; runs on any other machine are
harness checks, not measurements.** A run counts as a measurement only with `--device cuda` and the
`precise` profile, which these scripts always pass.

## Requirements

- A CUDA 12.8 capable NVIDIA driver (`nvidia-smi -L` must list the GPU) and `uv` on `PATH`.
- Disk for the Hugging Face cache (`paths.hf_home`): about 29 GB for the pinned Wan2.1-T2V-1.3B
  snapshot plus the perception weights of the `precise` profile.
- Docker is optional. When `docker info` responds, the Kubric image is pulled if missing and the
  `smoke` scenes are generated, simulated and perturbed so `kubric_render` can be timed; otherwise
  that target is skipped with its reason and the other targets are still measured.

## Command

```sh
bash scripts/h200/run_timing.sh
```

It runs, from `Repo/`:

1. `setup.sh`: `uv sync --no-default-groups --group dev --group cu128 --group perception --group
   generation`, a check that torch sees CUDA, `physverify extract fetch-weights` and
   `physverify timing fetch` (the pinned Wan2.1 snapshot; a cut download resumes).
2. `physverify timing env`: the environment of the run and whether it is a valid measurement.
3. The Kubric prerequisites (only when Docker responds, see above).
4. One `physverify timing run --dataset local_video --subset timing_wan21` with every target of
   `timing.order`. `wan21_generate` writes the generated videos and their `manifest.json` to
   `data/raw/local_video/timing_wan21/`; `extract_pipeline` and `verify_checks` then time those same
   clips.

Every `physverify` call gets `--profile precise --device cuda --set runtime.threads=N`, with `N` the
CPUs available to the job (`nproc`); set `PHYSVERIFY_TIMING_THREADS` to override it. The script
exits with the exit code of `timing run`. Without a CUDA GPU it exits with 2 before calling `uv`.

## Results

- `data/runs/<run_id>/timing/`: `results.json`, `samples.csv`, `stats.csv` and `summary.md`.
- `data/runs/<run_id>.log`: the console output of `timing run`.
- `data/runs/h200-timing-latest`: link to the latest run directory.

An interrupted run keeps the samples written so far. To rebuild the statistics and the summary:

```sh
uv run --no-sync physverify timing summarize --run data/runs/h200-timing-latest
```

## Budget

With a valid run, `timing budget` turns the measured per-clip costs into the number of training
prompts and pairs that fit in a GPU budget:

```sh
uv run --no-sync physverify timing budget --run data/runs/h200-timing-latest \
    --gpu-hours <H> --candidates <K> --eval-prompts <Q>
```

## Fine-tuning (ADR 0012)

Preference pairs from the verifier and a Flow-DPO LoRA of Wan2.1-T2V-1.3B. Every script needs a
CUDA GPU (exit 2 otherwise) and resumes after an interruption: run it again.

```sh
bash scripts/h200/smoke_finetune.sh                         # fast profile, end to end
PHYSVERIFY_FINETUNE_PILOT=1 bash scripts/h200/run_preference.sh   # pilot: 20 train prompts x K = 4
bash scripts/h200/run_preference.sh                          # all train and dev candidates
bash scripts/h200/run_finetune.sh                            # beta grid, training, selection
```

1. `smoke_finetune.sh`: the `fast` profile (campaign `wan21_smoke`, 2 prompts with one kept for
   val, K = 2) with the provisional scorer and relaxed pair rules; 5 training steps, a check that
   the val `kl_proxy` moved off 0, and one video generated with the trained adapter.
2. `run_preference.sh`: `finetune prompts build`, `generate` and `score` of the train (and dev)
   split, then `finetune pairs`. Stop after the pilot and review A8/A9 if pairs per prompt are
   below 0.3, abstention per candidate above 50 %, or fewer than 6 of 10 inspected pairs show the
   better video as winner.
3. `run_finetune.sh`: `finetune beta-grid` (4 short runs of a quarter of the main run each, chosen
   on val pairs), `finetune train` (warmup and checkpoint spacing are fractions of the run's
   steps; it exits 2 when the schedule would leave fewer checkpoints than `select` compares) and
   `finetune select` (dev scored for spaced checkpoints against the base).

`train` and `beta-grid` refuse pairs from the provisional scorer unless `--allow-provisional` is
passed, and the calibrated scorer is unavailable (exit 2) until A8's artifact has status
`calibrated`. `train --no-resume` refuses a run id that holds checkpoints unless `--force` is also
given, which deletes them.

Results live under `data/finetune/<campaign>/`: `prompts.jsonl`, `candidates/<model_tag>/`,
`scores/`, `pairs/<scorer>/` (`pairs.jsonl`, `val_pairs.jsonl`, `report.json`),
`train/<run_id>/` (`step-<n>/`, `metrics.csv`, `val.csv`, `run.json`) and `select/<run_id>/`.
`uv run --no-sync physverify finetune status` summarises a campaign, and
`physverify finetune budget --timing-run <dir> --gpu-hours <H>` turns a valid timing run with the
`dpo_step` target into the number of training prompts that fit.

## Machine-local root and environment

Every H200 path derives from one root, `PHYSVERIFY_H200_ROOT` (default `/home/mmonterot/work`,
set in `_common.sh`): the environment file `$PHYSVERIFY_H200_ROOT/env.sh`
(`PHYSVERIFY_H200_ENV` overrides it), the Kubric data directory, the Kubric lock and the udocker
binary. The environment file is not in git; its template is `env.example.sh`:

```sh
cp scripts/h200/env.example.sh "$PHYSVERIFY_H200_ROOT/env.sh"
source "$PHYSVERIFY_H200_ROOT/env.sh"   # HOME, XDG_CACHE_HOME, HF_HOME, UV_*, PATH, DD_TRACE_ENABLED
```

It derives every path from the directory it sits in. `run_kubric.sh`, `run_demo.sh`,
`run_h3.sh`, `setup_autoeval.sh` and `smoke_autoeval.sh` source it when
present (`source_h200_env` in `_common.sh`); source it by hand before the other scripts. The first
runs used the root `/tmp/mmonterot`, which the 30-day
purge of `/tmp` can wipe; `PHYSVERIFY_H200_ROOT=/tmp/mmonterot` still reproduces that layout.

## Kubric dataset (Docker or udocker)

The Kubric tasks run on CPU in the pinned image (`runtime.docker_image`), through Docker by
default or through [udocker](https://github.com/indigo-dc/udocker) in a named container created
from that same pinned digest when Docker is not available (`PHYSVERIFY_KUBRIC_ENGINE=udocker`;
the `[container]` section, see `configs/local.example.toml`). No GPU is needed.

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
- `PHYSVERIFY__PATHS__DATA_DIR` (default `$PHYSVERIFY_H200_ROOT/kubric-data`, outside `Repo/` so
  the laptop's `rsync --delete` cannot touch it);
- a Kubric-only heavy lock (`PHYSVERIFY_KUBRIC_LOCK`, default `$PHYSVERIFY_H200_ROOT/kubric.lock`,
  timeout 0): a second build on the same lock exits 75.

Preflight (exit 2 on failure): the data directory has at least `PHYSVERIFY_KUBRIC_MIN_FREE_GB`
(50) GB free; `physverify doctor` reports the chosen `container_engine`; then, for Docker,
`docker info` responds, the pinned image is pulled with `physverify dataset pull-image` when
`docker image inspect` misses it; for udocker, the binary exists and the container exists
(otherwise the image is pulled and the container created). The udocker image is
`runtime.docker_image` unless `PHYSVERIFY_KUBRIC_IMAGE` is set, and the image `udocker ps` lists for
the container must equal it: a container from another image (for example an old one created from
`kubricdockerhub/kubruntu:latest`) or one whose image cannot be read stops the script; remove it
with `udocker rm kub` so the next run recreates it from the pin, or set `PHYSVERIFY_KUBRIC_IMAGE`
to its image to use it knowingly. Last, `physverify doctor --probe-container` runs the configured
engine with the build's own mounts, workdir and environment and imports `kubric`, `bpy`,
`pybullet` and `physverify.kubric_worker.build_scene` from the mounted source. Then it runs
`physverify dataset build`
and `dataset list` with `--set runtime.threads=$PHYSVERIFY_KUBRIC_THREADS` (16, also Blender's
fixed thread count), `dataset.render.workers=$PHYSVERIFY_KUBRIC_WORKERS` (12) and
`dataset.render.jobs_per_process=$PHYSVERIFY_KUBRIC_JOBS` (8); render settings do not enter the
dataset hash. The log goes to `<data_dir>/runs/kubric-<subset>-<UTC time>.log`
(`kubric-latest.log` links to it) and starts with the engine and the image the container came
from; `dataset list` always runs, and the script exits with the build's exit code.

When a run fails or is cut, run the same command again: it resumes, and every step skips finished
items. If udocker is what failed and Docker responds, rerun it with
`PHYSVERIFY_KUBRIC_ENGINE=docker` (the default engine), keeping the same data directory.

## Demo (A12)

```sh
bash scripts/h200/run_demo.sh scenes      # after `h3 freeze` and the eval generation of run_h3.sh
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
never to the H3 one; `scenes` exits 2 until `h3/preregistration.json` exists or while a k = 0
video is missing. `scenes` scores with `finetune.scorer` only, never the provisional fallback, and
exits 2 when that scorer is unavailable or its label is not the `rule_label` frozen in the
preregistration (then refreeze). `run` and `report` fall back to the provisional rule, labelled as
such, while `demo.allow_provisional` is on. `physverify demo report --results <dir>` rebuilds a
report on CPU, for example once the calibrated rule replaces the provisional one.

## H3 comparison

```sh
source "$PHYSVERIFY_H200_ROOT/env.sh"
bash scripts/h200/run_h3.sh base-eval            # base eval videos; outcome-blind, can start first
bash scripts/h200/setup_autoeval.sh              # VideoPhy-2 AutoEval in its own venv (no GPU)
bash scripts/h200/smoke_autoeval.sh              # the gate: exit 1 when the pinned judge fails it
bash scripts/h200/run_h3.sh dev                  # AutoEval on dev videos (after `finetune select`)
bash scripts/h200/run_h3.sh freeze               # pre-registration, before any eval score
bash scripts/h200/run_h3.sh eval                 # ft videos, scores, eval judge rows, compare
bash scripts/h200/run_h3.sh status
```

ADR 0013 fixes the order. `freeze` writes `h3/preregistration.json`, and it must exist before any
eval score or judge row; `score`, `judge --split eval` and `compare` refuse to run without it.
When `h3 status` does not show `gate: passed`, `eval` skips the judge and still runs `compare`
(verdict at most `supported_verifier_only`); any other failure stops the script. The eval split
belongs to `h3`: `finetune generate|score --split eval` exit 2.
`base-eval` and `dev` accept the `physverify h3` options (`--shard I/N`, `--limit N`, `--task
pc|sa`). Every step skips the work it finds done with the same stamp.

`setup_autoeval.sh` is idempotent. It reads the pins from `[h3.autoeval]` in `configs/studies/generator_comparison.toml`
(upstream `code_commit`, `hf_repo` at `hf_revision`). Then it creates a Python 3.10 venv, checks the
upstream clone out at that commit and installs `autoeval_requirements.txt`. It downloads the model,
writes `checkpoint.sha256` and stores `uv pip freeze` in `freeze_file`, whose sha256 goes into the
gate report and the judge stamp. The relative paths sit under `PHYSVERIFY__H3__AUTOEVAL__ROOT`
(default `$PHYSVERIFY_H200_ROOT`). Only `transformers==4.28.1` is an upstream pin; the torch,
peft, numpy and Python versions are our choice, and the freeze file records the rest.
`PHYSVERIFY_AUTOEVAL_UPSTREAM` overrides the clone URL. The GPU steps exit 2 without a CUDA GPU
before any `uv` call.

## GT migrations

`physverify dataset migrate --name <name>` re-runs APPLY on every simulated clip of a stored
dataset in place (no version bump), keeping every render whose inputs did not change, and writes
`migrations/<name>.json`. `velocity-position` (the default, also `dataset migrate-velocity`) moves
a dataset to `velocity_convention = "position"`; the ground-truth fixes of `plan.GT_MIGRATIONS`
follow, in order: `floor-crossing` re-solves the floor bounces whose row after the crossing was
already under the floor, and `pair-collision` holds a ball's velocity at a ball-ball collision only
when a floor record overlaps the row's interval or the ball is within `contacts.tol_m` of the floor
(both need a dataset on `velocity-position`). A fix runs only as the first one the plan lacks and
covers every later one, since APPLY re-runs with the checkout's code. Afterwards rebuild every
listed subset (only the renders whose poses changed are redone) and `extract purge` the
re-rendered clips. A plan whose `gt_revision` is older than the checkout's is refused by every
build step until its pending fix (named in the refusal) has run on it.

## Perception campaign iterations (ADR 0015 amendment 2, WP-O)

```sh
source "$PHYSVERIFY_H200_ROOT/env.sh"
PHYSVERIFY_ITER_ID=it0-baseline PHYSVERIFY_ITER_EXPECT_COMMIT=<head> \
    PHYSVERIFY_ITER_CACHE_DIR=<shared cache root> \
    nohup bash scripts/h200/run_iteration.sh > "$PHYSVERIFY_H200_ROOT/iterations/it0.log" 2>&1 &
```

`run_iteration.sh [FROM [TO]]` runs, on the `kubric_synthetic` calib clips only: `extract run`,
`diagnose stages --write-profile` (its own error profile), `extract run` again, `calibrate
stage-sigma` and `calibrate devscore --level both`, the development metric (H1 of the full calib
fit and J of out-of-fold decisions per check, AUROC of the clip score out of fold). Everything
lands in `$PHYSVERIFY_H200_ROOT/iterations/<id>-<device>/` (calibration directory, runs, heavy
lock, `devscore.json`, the full calib fit `calibration.json`, `summary.md`); the header lists the
startup checks and variables. `PHYSVERIFY_ITER_PROFILE` (default `precise`) picks the sampling
profile, which also fixes the dataset version the calib clips are listed from. With
`PHYSVERIFY_ITER_DEVICE=cuda` launch it through `gpu_queue.sh` with one GPU,
`CUDA_VISIBLE_DEVICES` in 0-3 (anything else, a list included, exits 2).

Declared arms (it3): `PHYSVERIFY_ITER_ARM=<arm>` names a line of `iteration_arms.tsv`, whose
provider swaps (`extraction.plan.<stage>=<provider>` only) go as `--set` to every physverify
call. The profile stays the committed one, so arms that share a provider prefix share its cache
keys and its error-profile entries (a separate profile name would not: the name is part of every
entry). `doctor` must list each swap among its sources, `state.env` records `CHAIN_ARM` (a
resume with another arm exits 2) and the summary names the arm and the GPU. Give each arm its
own `PHYSVERIFY_ITER_ID` (e.g. `it3-c`), so it lands in `iterations/it3-c-cuda/`.

The sealed ledger (`configs/verifier/sealed.toml`): `calibrate fit|stage-sigma|evaluate|devscore`, `abstain
fit|curve`, `verify`, `diagnose *` and `scene-stats per-clip` exit 2 before computing on a sealed
set (`kubric_synthetic:eval`, the fresh controls) unless `PHYSVERIFY_UNSEAL` names it, and then
append a line to the ledger first. A look never creates the ledger: create it once, by hand,
with `env.sh` sourced (`touch "$PHYSVERIFY_H200_ROOT/sealed-ledger.txt"`); a missing file (a
wrong `HOME`, another path) stops the look, and stderr prints the path it resolved. A look needs
a clean git checkout (nothing uncommitted under `src` or `configs`). Its frozen state is the
commit, the configuration sections with profile and device (not paths, threads or the lock) and
the sha256 of `calibration.json` (or the `--artifact` used), `thresholds.json`,
`exclusions.json` and the error profile; a look under another frozen state, a refit at the same
commit included, also needs `PHYSVERIFY_UNSEAL_REPEAT=1`. A missing `[sealed]` section fails
closed.

## Freeze and final looks (amendment 7)

Both run in a clean checkout at the freeze commit `C_F`, on the CPU, profile `precise`, with the
committed configs only; each header lists its startup checks and variables (every refusal exits 2
before any physverify call).

```sh
source "$PHYSVERIFY_H200_ROOT/env.sh"
PHYSVERIFY_FREEZE_EXPECT_COMMIT=$C PHYSVERIFY_FREEZE_ITERATION=<J4 choice> \
    PHYSVERIFY_FREEZE_ITERATIONS="<iteration dirs>" PHYSVERIFY_FREEZE_P2_COMMIT=<P2 commit> \
    PHYSVERIFY_FREEZE_V1_COMMIT=<C_V1> PHYSVERIFY_FREEZE_V1_CALIBRATION=<base calibration.json> \
    PHYSVERIFY_FREEZE_CACHE_DIR=<cache> \
    nohup bash scripts/h200/run_freeze.sh > "$PHYSVERIFY_H200_ROOT/final/freeze.log" 2>&1 &
```

`run_freeze.sh [FROM [TO]]` adopts the iteration J4 chooses (`freeze_query.py select`; never
refitted), extracts the development real video, runs the rule 1.5 evaluation on calib, the two R2
arms, `exclusions decide`, `abstain fit`, and writes `$F/freeze.json` (`freeze_query.py
manifest`: commit, configs tree, artifact sha256, `record_sha`, the seed of amendment 2 section 2,
the J3 and J13 diffs and the v1 arm's commit and calibration). `$F` defaults to
`$PHYSVERIFY_H200_ROOT/final`.

```sh
export PHYSVERIFY_FINAL_EXPECT_COMMIT=$C PHYSVERIFY_FINAL_SEED=<amendment seed>
bash scripts/h200/run_final.sh seed
bash scripts/h200/run_final.sh render kubric_fresh          # then kubric_fresh_ood
bash scripts/h200/run_final.sh fetch physics_iq_views       # and objectron_static
bash scripts/h200/run_final.sh extract kubric_fresh         # each set, real_new as its two sets
bash scripts/h200/run_final.sh look kubric_fresh            # once per set; real_new together
bash scripts/h200/run_final.sh v1arm kubric_fresh           # kubric_synthetic_eval, real_new
```

`run_final.sh` checks the freeze (commit, device, profile, calibration directory, frozen files,
seed) on every command, refuses a set whose cache existed before its first extraction (J1), a
Kubric set without the render of the seed, a render after a look (J2), a second look at any set
the ledger covers (J6; `look SET --retry` once, when that look wrote no evaluation), and changed
downloads (J14). Bookkeeping, logs and per-command summaries land in `$F/run-final/`; commands on
different sets may run at the same time (each holds its own locks; the heavy lock is
`$F/heavy.lock`). The v1 arm runs at `C_V1` through `overlay_cli.py` with the freeze's configs and
the base `checks.toml`, the base calibration and `PHYSVERIFY_UNSEAL_REPEAT=1`, and must leave no
`meta.json` in its overlay. `PHYSVERIFY_FINAL_DRYRUN=1` is the rehearsal: `$F`, the data, the
cache and its own ledger (`PHYSVERIFY__SEALED__LEDGER`) under `$PHYSVERIFY_H200_ROOT/scratch`,
a seed other than the freeze's, never `kubric_synthetic_eval`, and, for the new real control,
`PHYSVERIFY__PHYSICS_IQ_VIEWS__VIEWS='["center"]'` and `PHYSVERIFY__OBJECTRON_STATIC__SPLIT=train`.
A run with `PHYSVERIFY_FINAL_EXTRACT_LIMIT` (the pace measurement) is recorded as
`extract-limited`: a look still needs a full `extract`.

Look order (plan-congelado, "Recortes"; the script enforces only that a v1 arm follows its look
and that no render follows a look, so this order is kept by hand): (1) `look
kubric_synthetic_eval` (the set already seen), (2) `look kubric_fresh`, (3) `run_h2.sh dev`, then
`test`, (4) `look real_new`; afterwards, with the same freeze and seed, the `kubric_fresh_ood`
render and look, `look kubric_fresh_blind` and the v1 arms.

H2 then runs from the freeze manifest instead of the A8 v2 run:
`PHYSVERIFY_H2_FREEZE=$F/freeze.json PHYSVERIFY_H2_CALIBRATION_DIR=$F/calibration bash
scripts/h200/run_h2.sh dev` (and `test` once).

Debt: `run_h2.sh` keeps its own startup guards (not `_guards.sh`).
