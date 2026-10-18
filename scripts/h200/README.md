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

## Kubric dataset (udocker)

The H200 has no Docker and no root, so the Kubric tasks run through
[udocker](https://github.com/indigo-dc/udocker) in a named container created from
`kubricdockerhub/kubruntu` (`[container] engine = "udocker"`; see `configs/local.example.toml`).
No GPU is needed.

```sh
# smoke: fast profile, 2 clips, fresh data directory
PHYSVERIFY_KUBRIC_PROFILE=fast PHYSVERIFY__PATHS__DATA_DIR=/tmp/mmonterot/kubric-smoke \
    bash scripts/h200/run_kubric.sh smoke
# full dataset, precise profile, in the background
nohup bash scripts/h200/run_kubric.sh full >/dev/null 2>&1 &
```

`run_kubric.sh [subset]` (default `full`) sources `/tmp/mmonterot/env.sh` when present
(`PHYSVERIFY_H200_ENV` overrides the path) and exports:

- the four `DD_*=false` variables, without which the host Python's ddtrace breaks udocker;
- `PHYSVERIFY__CONTAINER__ENGINE=udocker`, `..._UDOCKER_BIN` and `..._UDOCKER_CONTAINER`;
- `PHYSVERIFY__PATHS__DATA_DIR` (default `/tmp/mmonterot/kubric-data`, outside `Repo/` so the
  laptop's `rsync --delete` cannot touch it);
- a Kubric-only heavy lock (`/tmp/mmonterot/kubric.lock`, timeout 0): a second build exits 75.

Preflight (exit 2 on failure): the udocker binary exists; the container exists, otherwise it is
pulled and created; it imports `kubric`, `bpy` and `pybullet`; the data directory has at least
`PHYSVERIFY_KUBRIC_MIN_FREE_GB` (50) GB free; `physverify doctor` reports
`container_engine: udocker`. Then it runs `physverify dataset build` and `dataset list` with
`--set runtime.threads=$PHYSVERIFY_KUBRIC_THREADS` (16, also Blender's fixed thread count),
`dataset.render.workers=$PHYSVERIFY_KUBRIC_WORKERS` (12) and
`dataset.render.jobs_per_process=$PHYSVERIFY_KUBRIC_JOBS` (8); render settings do not enter the
dataset hash. The log goes to `<data_dir>/runs/kubric-<subset>-<UTC time>.log`
(`kubric-latest.log` links to it), `dataset list` always runs, and the script exits with the
build's exit code. Running it again resumes: every step skips finished items.
