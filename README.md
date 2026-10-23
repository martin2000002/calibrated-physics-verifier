# physverify

Calibrated physics verifier for generated video of rigid bodies. Given a clip, `physverify`
extracts object states and contacts (from ground truth or from perception models), evaluates
physics checks R1-R5b (interpenetration, support equilibrium, collision energy, object integrity,
free-fall shape and gravity value) and reports each verdict with the propagated measurement error,
abstaining when the error is too large to decide.

It is part of an undergraduate thesis (Computer Science, USFQ) built on three studies: the
calibration of each check against violations of known size injected into synthetic scenes, the
agreement of the verifier with human physics labels on generated video (VideoPhy) compared with a
reference vision-language model, and the comparison of a video generator fine-tuned with the
verifier as preference signal against best-of-8 sampling from the base generator.

## Status

Every subcommand is implemented and covered by unit tests. The checks are calibrated on the
calibration split of the synthetic dataset only; `calibrate publish` writes the thresholds that
`verify` uses, and the values in `configs/verifier/checks.toml` are provisional fallbacks for when
none is published. The tag `final-freeze` marks the frozen verifier (code and configuration) under
which the sealed sets of `configs/verifier/sealed.toml` are evaluated. A command refuses to compute
on a sealed set unless it is unsealed explicitly, and each look is logged before any number is
computed.

The calibration artifact is a run output and is not versioned. The size model is
(`configs/verifier/size_model.toml`): its coefficients are estimates and carry their measured
error. Generation, fine-tuning and the generator comparison need a CUDA GPU; the demo generates a
video on the GPU or scores a given video on the CPU.

| Command | What it does |
|---|---|
| `timing` | Per-clip cost harness (generation, extraction, checks, rendering, DPO step) and GPU budget; only GPU runs with `--device cuda --profile precise` count as measurements |
| `synth` | Kubric synthetic dataset: five scene families, ten perturbation types at four magnitude levels, per-stage ground truth; resumable steps and ground-truth migrations |
| `extract` | Eight-stage perception pipeline (detection, tracking, depth, camera, 3D lifting, scale with provenance, kinematics, contacts) with `fast` (CPU) and `precise` (GPU) profiles; weight download and cache purge |
| `diagnose` | Extractor error per stage, end to end and by factorial attribution, plus real video (Physics-IQ, PISA) |
| `download` | Real datasets (Physics-IQ, PISA) and, by name, the sealed real controls |
| `verify` | Checks R1-R5b (nine measures) on ground-truth ("oracle") or perceived states, with verdict intervals, abstention and a specificity sweep |
| `calibrate` | Calibration on the calibration split: thresholds, false-positive rates, detection limits, per-check and per-stage sigma correction, clip score; held-out and real-video evaluation; publication of the thresholds |
| `exclusions` | Applies the pre-registered exclusion rules to the calibration results and writes the hashed decision record the later studies read |
| `selective-risk` | Risk-coverage of the threshold policy: abstention level chosen on the calibration split, held-out curve |
| `size` | Per-class object size model learned on Objectron: fetch, fit, held-out and end-to-end error |
| `agreement` | Verifier vs reference VLM on VideoPhy human labels: data, reviewed subset, pre-registration on dev, comparison on test |
| `finetune` | Prompt set, Wan2.1-T2V-1.3B candidates scored by the verifier, preference pairs with a pilot stop rule, Flow-DPO LoRA training and checkpoint selection |
| `compare-generators` | Fine-tuned vs base vs best-of-8 under the verifier and VideoPhy-2 AutoEval, with a pre-registration and a regression gate for the judge |
| `demo` | One prompt or video through the whole verifier: per-check report and overlay video |
| `scene-stats` | Per-clip detection and depth errors, per-scene tables and scene-bootstrap differences between two extractor configurations |
| `results` | Collects the reported numbers of a calibration chain (the stored output directory of a calibration-chain runner that is no longer in the repository) and an agreement run into `results.md` and `results.json`, each value with its source and estimates marked |
| `doctor` | Effective configuration, paths, lock and package versions |

Runs on the university H200 use the scripts in `scripts/h200/`: on the GPU, timing, fine-tuning,
the generator comparison and demo generation; on the CPU, Kubric rendering, the perception
iterations (CUDA optional), the freeze, the final looks and the human-agreement study. See
[its README](scripts/h200/README.md).

## Installation

Requires Linux, [uv](https://docs.astral.sh/uv/) (Python 3.12) and, for synthetic data, Docker or
udocker with the pinned Kubric image (`runtime.docker_image` in `configs/default.toml`; the engine
is set in the `[container]` table, see `configs/local.example.toml`).

```sh
uv sync                                   # default groups: dev, cpu (CPU PyTorch), perception, analysis, data
uv sync --no-default-groups --group dev   # light: core + dev tools only
uv sync --no-default-groups --group dev --group cu128 --group perception --group generation --group data   # CUDA 12.8
cp configs/local.example.toml configs/local.toml   # machine-local overrides (ignored by git)
```

`generation` (diffusers, peft) is only needed for Wan2.1 generation and fine-tuning. `cpu` and
`cu128` are conflicting groups with their own PyTorch indexes; `uv.lock` covers both. Use `uv run
--no-sync` so uv does not install the default groups behind your back; installs are heavy jobs and
run under the lock (`flock <lock> uv sync ...`).

## Usage

```sh
uv run --no-sync physverify --help
uv run --no-sync physverify <command> --help
uv run --no-sync physverify doctor
uv run --no-sync physverify --profile precise --device cuda doctor
uv run --no-sync physverify --set runtime.threads=2 doctor
```

Global options go before the subcommand: `--config-dir` (default `$PHYSVERIFY_CONFIG_DIR` or
`configs/`), `--profile fast|precise`, `--device cpu|cuda|cuda:N`, `--set section.key=value`
(repeatable), `--no-lock`, `--lock-timeout SECONDS`, `--log-level`. Configuration is layered,
lowest to highest: model defaults, `configs/**/*.toml` (except the root's `local*.toml`), the
active profile's `[<section>.profiles.<name>]` tables, `configs/local.toml`,
`PHYSVERIFY__SECTION__KEY` environment variables, `--device` / `--profile`, then `--set`. `doctor`
prints every effective value with its source.

`configs/` holds `default.toml` and `local.example.toml`; `verifier/` the verifier's own settings
(checks, extraction, calibration, sizing, sealed sets), `datasets/` one file per dataset and
`studies/` one file per study. Each top-level table lives in exactly one file.

Exit codes: 0 ok, 1 error, 2 usage, configuration or resource budget, 75 heavy lock busy.

### Renamed names

Older runs, notes and launchers may use these names; the old names are no longer read.

| Old | Current |
|---|---|
| `physverify h3` | `physverify compare-generators` |
| `physverify abstain` | `physverify selective-risk` |
| `physverify data` | `physverify download` |
| `physverify dataset` | `physverify synth` |
| `physverify calibrate devscore` | `physverify calibrate dev-score` |
| `results` options `--a8v2-dir`, `--h1-evaluation`, `--h2-run`, `--abstention-run` | `--chain-dir`, `--calibration-evaluation`, `--agreement-run`, `--selective-risk-run` |
| `PHYSVERIFY_H2_*` | `PHYSVERIFY_AGREEMENT_*` |
| `PHYSVERIFY_H2_A8V2_DIR` | `PHYSVERIFY_AGREEMENT_CHAIN_DIR` |
| `PHYSVERIFY_ITER_*` | `PHYSVERIFY_ITERATION_*` |
| `PHYSVERIFY_FREEZE_P2_COMMIT` | `PHYSVERIFY_FREEZE_EARLY_EXTRACTION_COMMIT` |
| `scripts/h200/setup.sh` | `scripts/h200/setup_cuda.sh` |
| `scripts/h200/run_h2.sh`, `h2_query.py` | `scripts/h200/run_human_agreement.sh`, `agreement_query.py` |
| `scripts/h200/run_h3.sh` | `scripts/h200/run_generator_comparison.sh` |
| `scripts/h200/run_iteration.sh`, `iteration_arms.tsv` | `scripts/h200/run_perception_iteration.sh`, `perception_arms.tsv` |
| `scripts/h200/a8v2_query.py` | `scripts/h200/run_query.py` |
| `scripts/h200/run_a8v2.sh` (`PHYSVERIFY_A8V2_*`) | Removed. Its stored output directory, a calibration chain, is still read by `results --chain-dir` and `run_human_agreement.sh`; new runs use `run_freeze.sh` and its `freeze.json` |
| `scripts/h200/run_p1_perception.sh` (`PHYSVERIFY_P1_*`) | Removed. Perception configurations run as declared arms of `scripts/h200/run_perception_iteration.sh` (`perception_arms.tsv`) |
| `scripts/h200/migrate_velocity.sh` (`PHYSVERIFY_MIGRATE_*`) | Removed. `physverify synth migrate`, then `synth build` and `extract purge` by hand (see the H200 README, Ground-truth migrations) |

Persisted names (record classes, schema ids, config tables and keys, ledger labels) did not
change, so existing runs, caches and artifacts stay readable.

## Development

```sh
uv run --no-sync pytest                 # unit tests; no network, GPU or Docker
uv run --no-sync pytest -m docker       # Python 3.9 container boundary (needs the Kubric image)
uv run --no-sync ruff check . && uv run --no-sync ruff format --check .
uv run --no-sync mypy
```

`CLAUDE.md` describes the layers, how to add a plugin and the contracts that must not change.

## Package map

The code lives in `src/physverify/`: frozen wire records shared with the Python 3.9 container
(`schema`) and its entry point (`kubric_worker`), shared infrastructure (`core`, `stats`,
`config`, `runtime`), the physics and the data (`physics`, `synth`, `sizing`, `datasets`), the
verifier itself (`extraction`, `checks`, `verification`), the Wan2.1 generator (`generation`), the
studies and tools built on them (`diagnosis`, `calibration`, `human_agreement`, `selective_risk`,
`finetune`, `freeze`, `generator_comparison`, `timing`, `demo`, `reporting`), the composition root
`app` and the command line `cli`. The package tree, one line per package, and the layer rule are
in `CLAUDE.md`.

Data, model weights and run outputs live under `data/` (ignored by git).
