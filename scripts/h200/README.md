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
