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
`run_h3.sh`, `run_p1_perception.sh`, `setup_autoeval.sh` and `smoke_autoeval.sh` source it when
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

## P1 perception

```sh
source "$PHYSVERIFY_H200_ROOT/env.sh"
CUDA_VISIBLE_DEVICES=1 PHYSVERIFY_TIMING_THREADS=32 \
    nohup bash scripts/h200/run_p1_perception.sh v1 > "$PHYSVERIFY_H200_ROOT/p1/p1.log" 2>&1 &
PHYSVERIFY_P1_CLIPS="<clip ids>" bash scripts/h200/run_p1_perception.sh v1-precise A-fast <ID>
```

Runs the P1 depth (A) and detection (B) configurations of ADR 0014 (IDs `A-fast A-518 A-280 A-392
A-e2 A-met B-tpq B-bj B-bpq`; default: all but `A-392`). `A-fast` and the B rows use the `fast`
profile, the other A rows `precise`. Every option is a root `--set` placed before the subcommand:
`runtime.threads`, `runtime.heavy_lock=$P1_DATA/p1.lock`, `paths.data_dir=$P1_DATA`,
`paths.hf_home=$P1_HF`, a per-configuration `paths.error_profile` (written by `diagnose stages
--write-profile` from `calib`, read by every later step), the dataset version (`v1`: 128 px, plain;
`v1-precise`: 256 px, HDRI) and the row's own settings. The selection is `--subset local` on `v1`, `--subset full` (or `PHYSVERIFY_P1_CLIPS`) on
`v1-precise`.

`P1_DATA` is `PHYSVERIFY_P1_DATA` (default `$PHYSVERIFY_H200_ROOT/p1/data`) and `P1_HF` is
`PHYSVERIFY_P1_HF` (default `$PHYSVERIFY_H200_ROOT/p1/hf`). The first run seeds them, reading only:
`synthetic`, `weights` and `vendor` from `$PHYSVERIFY_H200_ROOT/Repo/data`, the whole HF `hub/`
(hard links when possible) and `thresholds.perceived.fast.json`, pinned as
`$P1_DATA/calibration/thresholds.json`; existing copies are kept. On `v1-precise` it links
`$P1_DATA/synthetic/v1-precise` to `$PHYSVERIFY_H200_ROOT/kubric-data/synthetic/v1-precise`, since
manifest video paths are relative to `paths.data_dir`. A pinned table with no `checks` stops the
batch (every verdict would be UNSCORED) unless `PHYSVERIFY_P1_PROVISIONAL=1`, which points
`paths.thresholds` at a missing file so verify falls back to `configs/checks.toml`, and records
`provisional:<sha of checks.toml>` in the manifest. Per configuration it runs
`extract fetch-weights`, `extract run`, `diagnose stages`, `scene-stats per-clip` (to
`$P1_DATA/runs/p1-<set>-<ID>.json`; compare two with `scene-stats delta`), `diagnose end-to-end`, `verify --level perceived`
and, with `PHYSVERIFY_P1_ATTRIBUTION=1`, `diagnose attribution`. Exit 1 (some clips failed) is
reported and tolerated; any other failure stops the batch. Logs go to `$P1_DATA/runs/p1-logs/`, and
`$P1_DATA/runs/p1-manifest.tsv` records per step the run directory, `CUDA_VISIBLE_DEVICES`, the
sha256 of `src/**/*.py` and of the thresholds. Exit 2 without a CUDA GPU, before any `uv`.

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

`setup_autoeval.sh` is idempotent. It reads the pins from `[h3.autoeval]` in `configs/h3.toml`
(upstream `code_commit`, `hf_repo` at `hf_revision`). Then it creates a Python 3.10 venv, checks the
upstream clone out at that commit and installs `autoeval_requirements.txt`. It downloads the model,
writes `checkpoint.sha256` and stores `uv pip freeze` in `freeze_file`, whose sha256 goes into the
gate report and the judge stamp. The relative paths sit under `PHYSVERIFY__H3__AUTOEVAL__ROOT`
(default `$PHYSVERIFY_H200_ROOT`). Only `transformers==4.28.1` is an upstream pin; the torch,
peft, numpy and Python versions are our choice, and the freeze file records the rest.
`PHYSVERIFY_AUTOEVAL_UPSTREAM` overrides the clone URL. The GPU steps exit 2 without a CUDA GPU
before any `uv` call.

## GT migrations (ADR 0004: WP-G 1.C velocity, WP-H floor crossing, WP-I pair collision)

`migrate_velocity.sh` runs one GT migration of a stored dataset in place (CPU only, no version
bump), picked by `PHYSVERIFY_MIGRATE_NAME`: `velocity-position` (default) moves it to
`velocity_convention = "position"`; the ground-truth fixes of `plan.GT_MIGRATIONS` follow, in
order: `floor-crossing` re-solves the floor bounces whose row after the crossing was already under
the floor, and `pair-collision` holds a ball's velocity at a ball-ball collision only when a floor
record overlaps the row's interval or the ball is within `contacts.tol_m` of the floor (both need
a dataset on `velocity-position`). A fix runs only as the first one the plan lacks and covers every
later one, since APPLY re-runs with the checkout's code. Each
run is `dataset migrate --name <name>`, a rebuild of every listed subset (only the renders whose
poses changed are redone), `extract purge` of the re-rendered clips on every cache root, oracle
sweeps with `fit_source` position and velocity, and `dataset audit-migration --name <name>`
against a copy taken before. Every written path must lie under `PHYSVERIFY_MIGRATE_ROOT`; the
header lists the variables and the commands for a dev copy and for the full v1-precise dataset.
A plan whose `gt_revision` is older than the checkout's is refused by every build step until its
pending fix (named in the refusal) has run on it.

It holds the Kubric lock (`PHYSVERIFY_MIGRATE_LOCK`, default `${PHYSVERIFY_KUBRIC_LOCK:-$ROOT/kubric.lock}`),
so it never renders beside a `run_kubric.sh` build. It stops after `dataset migrate` when any clip
changed state (applied or skipped) until `PHYSVERIFY_MIGRATE_ALLOW_STATUS_CHANGES=1`, and moves the
calibration artifacts fitted on the old GT into `<calibration dir>/stale-<name>/` once,
printing a notice; recalibrate afterwards. `calibration-baseline-*` directories are refused.

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
startup checks and variables. With `PHYSVERIFY_ITER_DEVICE=cuda` launch it through `gpu_queue.sh`
(GPUs 0-3).

The sealed ledger (`configs/sealed.toml`): `calibrate fit|stage-sigma|evaluate|devscore`, `abstain
fit|curve`, `verify`, `diagnose *` and `scene-stats per-clip` exit 2 before computing on a sealed
set (`kubric_synthetic:eval`, the fresh controls) unless `PHYSVERIFY_UNSEAL` names it, and then
append a line to `$PHYSVERIFY_H200_ROOT/sealed-ledger.txt` (with `env.sh` sourced) first. A look
under another frozen state (commit, config, thresholds, error profile) also needs
`PHYSVERIFY_UNSEAL_REPEAT=1`. A baseline rerun of `run_a8v2.sh` on main therefore stops at its
first look at eval (an eval preflight clip, or step 2: `diagnose stages --subset full` reports on
eval) unless `PHYSVERIFY_UNSEAL=kubric_synthetic:eval` is exported by hand, which records each of
its looks; the checkout of the running chain is not updated until it finishes.

Debt: `_steps.sh` `require_completed` repeats `run_a8v2.sh`'s `require_earlier_steps`; it goes
away when `run_a8v2.sh` moves to `_steps.sh` after the A8 v2 chain completes.
