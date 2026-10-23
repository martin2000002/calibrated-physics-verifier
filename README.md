# physverify

Calibrated physics verifier for generated video of rigid bodies. Given a clip, `physverify`
extracts object states and contacts (from ground truth or from perception models), evaluates
physics checks R1-R5 (interpenetration, support equilibrium, collision energy, object integrity,
free-fall shape and gravity value) and reports each verdict with the propagated measurement error,
abstaining when the error is too large to decide.

Part of an undergraduate thesis (Computer Science, USFQ) that tests three hypotheses: each check's
magnitude grows monotonically with the size of an injected violation, with a measured threshold and
false-positive rate (H1); on human-labelled generated video (VideoPhy), the verifier agrees with
the human labels better than a reference VLM (H2); a generator fine-tuned with the verifier as a
preference signal beats best-of-8 sampling under the same verifier (H3). The package map below
describes the architecture.

## Status

Every subcommand is implemented and covered by unit tests. The checks are calibrated on the
calibration split of the synthetic dataset only; `calibrate publish` writes the thresholds that
`verify` uses, and the values in `configs/checks.toml` are provisional fallbacks for when none is
published. The tag `final-freeze` marks the frozen verifier (code and configuration) under which
the sealed sets of `configs/sealed.toml` and the H2 comparison are evaluated. A command refuses to
compute on a sealed set unless it is unsealed explicitly, and each look is logged before any
number is computed.

The calibration artifact is a run output and is not versioned. The size model is
(`configs/size_model.toml`): its coefficients are estimates and carry their measured error.
Fine-tuning (A11) and the H3 comparison (A13) are implemented; their generation and training steps
need a CUDA GPU. The demo (A12) generates a video on the GPU or scores a given video on the CPU.

| Command | What it does | Activity |
|---|---|---|
| `timing` | Per-clip cost harness (generation, extraction, checks, rendering, DPO step) and GPU budget; only H200 runs (`--device cuda --profile precise`) count as measurements | A3 |
| `dataset` | Kubric synthetic dataset: five scene families, ten perturbation types at four magnitude levels, per-stage ground truth; resumable steps and ground-truth migrations | A4 |
| `extract` | Eight-stage perception pipeline (detection, tracking, depth, camera, 3D lifting, scale with provenance, kinematics, contacts), `fast` (CPU) and `precise` (GPU) profiles; weight download | A5 |
| `diagnose` | Extractor error per stage, end to end and by factorial attribution, plus real video (Physics-IQ, PISA) | A6 |
| `data` | Download of the real datasets (Physics-IQ, PISA) and, by name, of the sealed real controls | A6 |
| `verify` | Checks R1-R5b (nine measures) on ground-truth ("oracle") or perceived states, with verdict intervals, abstention and a specificity sweep | A7 |
| `calibrate` | H1 calibration on the calibration split: thresholds, false-positive rates, detection limits, per-check and per-stage sigma correction, clip score; held-out and real-video evaluation; publication of the thresholds | A8 |
| `exclusions` | Applies the pre-registered exclusion rules to the calibration results; writes the hashed decision record that H2 and H3 read | A8 |
| `abstain` | Risk-coverage of the threshold policy: choice of the abstention level k* on the calibration split, held-out curve | A9 |
| `size` | Per-class object size model learned on Objectron: fetch, fit, held-out and end-to-end error | A9 |
| `agreement` | Verifier vs reference VLM on VideoPhy human labels (H2): data, reviewed subset, pre-registration on dev, comparison on test | A10 |
| `finetune` | Prompt set, Wan2.1-T2V-1.3B candidates scored by the verifier, preference pairs with a pilot stop rule, Flow-DPO LoRA training and checkpoint selection | A11 |
| `demo` | One prompt or video through the whole verifier: per-check report and overlay video | A12 |
| `h3` | Fine-tuned vs base vs best-of-8 under the verifier and VideoPhy-2 AutoEval (H3), with a pre-registration and a regression gate for the judge | A13 |
| `scene-stats` | Per-clip detection and depth errors, per-scene tables and scene-bootstrap differences between two extractor configurations | - |
| `results` | Collects the reported numbers of a calibration chain (and an H2 run) into `results.md` and `results.json`, each value with its source and estimates marked | - |
| `doctor` | Effective configuration, paths, lock and package versions | - |

GPU runs (Kubric rendering, the `precise` profile, timing, fine-tuning, H3, demo generation) use
the scripts in `scripts/h200/`; see [its README](scripts/h200/README.md).

## Installation

Requires Linux, [uv](https://docs.astral.sh/uv/) (Python 3.12) and, for synthetic data, Docker or
udocker with the pinned Kubric image (`runtime.docker_image` in `configs/default.toml`; the engine
is set in the `[container]` section, see `configs/local.example.toml`).

```sh
uv sync                                   # default groups: dev, cpu (CPU PyTorch), perception, analysis, data
uv sync --no-default-groups --group dev   # light: core + dev tools only
uv sync --no-default-groups --group dev --group cu128 --group perception --group generation --group data   # CUDA 12.8 machines
cp configs/local.example.toml configs/local.toml   # machine-local overrides (ignored by git)
```

`generation` (diffusers, peft) is only needed for Wan2.1 generation and fine-tuning.

## Usage

```sh
uv run --no-sync physverify --help
uv run --no-sync physverify <command> --help
uv run --no-sync physverify doctor
uv run --no-sync physverify --profile precise --device cuda doctor
uv run --no-sync physverify --set runtime.threads=2 doctor
```

Global options go before the subcommand. Configuration is layered: model defaults,
`configs/*.toml`, the active profile, `configs/local.toml`, `PHYSVERIFY__SECTION__KEY` environment
variables, `--device` / `--profile`, then `--set`. `doctor` prints every effective value with its
source.

Exit codes: 0 ok, 1 error, 2 usage, configuration or resource budget, 75 heavy lock busy.

## Development

```sh
uv run --no-sync pytest                 # unit tests; no network, GPU or Docker
uv run --no-sync pytest -m docker       # Python 3.9 container boundary
uv run --no-sync ruff check . && uv run --no-sync ruff format --check .
uv run --no-sync mypy
```

## Package map

| Package | Role |
|---|---|
| `physverify.schema` | Frozen records shared by host (3.12) and container (3.9): clips, sampling, scenes, states, contacts, perception outputs, scale, error profiles, check results; JSON and NPZ codecs |
| `physverify.core` | Plugin discovery, hashing and cache keys, errors and exit codes, logging, resampling, video decoding |
| `physverify.config` | Settings models and the layered TOML loader |
| `physverify.runtime` | Heavy-job lock, container runners (Docker, udocker), thread limits, run directories |
| `physverify.physics` | Geometry (signed distance, contact manifolds, support polygons), finite differences, kinematics with closed-form sigma, static equilibrium, error propagation and the numeric definition of every checked quantity |
| `physverify.kubric_worker` | Container entry point and tasks (Kubric, PyBullet, Blender) |
| `physverify.datasets` | Dataset, video and ground-truth protocols; plugins for the Kubric sets, local videos, Physics-IQ, PISA, VideoPhy and the sealed real controls |
| `physverify.synth` | Synthetic scene families, perturbations, ground-truth export and dataset migrations |
| `physverify.extraction` | Stage providers (perceived and ground truth), stage cache and the extraction pipeline |
| `physverify.checks` | Physics checks R1-R5b |
| `physverify.verification` | Check inputs, runner, verdict policies, specificity sweep, risk-coverage and the check scope of H2 and H3 |
| `physverify.diagnosis` | Per-stage metrics, end-to-end error, attribution, real-video diagnosis and per-scene statistics |
| `physverify.calibration` | H1 calibration: thresholds, rates, detection limits, sigma recalibration, weights and clip score; held-out evaluation, exclusion rules and the sealed-set ledger |
| `physverify.sizing` | Object size model learned on Objectron: annotation download, fit and evaluation |
| `physverify.agreement` | H2 comparison: verifier decisions, agreement statistics, pre-registration and report |
| `physverify.finetune` | Preference fine-tuning of Wan2.1: prompts, generation, scoring, pairs, stop rule, Flow-DPO trainer, checkpoint selection and GPU budget |
| `physverify.evaluation` | H3 comparison: arms, judges, pre-registration, statistics and verdict |
| `physverify.demo` | End-to-end demo: report and overlay video for one prompt or video |
| `physverify.reporting` | Results report: reported numbers read from run outputs, each with its source |
| `physverify.timing` | Timing harness, targets, the Wan2.1 generator call and GPU budget |
| `physverify.cli` | `physverify` command line; subcommands are discovered from `cli/commands/` |

Data, model weights and run outputs live under `data/` (ignored by git).

## Adding a check

A check is one file, `src/physverify/checks/<id>_<name>.py`, that exports a `PLUGINS` tuple. Each class in it
satisfies `physverify.checks._base.Check`: a `check_id`, the `requires` set of inputs it needs, its
`magnitude_unit`, a pydantic `Params` model with its defaults, and `evaluate(inputs)`. Defaults can be overridden
in a `[checks.<id>]` table of `configs/checks.toml`. The registry discovers the file on its own; no other module
changes. Scene families, perturbations, datasets, stage providers, diagnosis metrics, verdict policies, clip
scorers, timing targets and subcommands are added the same way, one file in their own package.
