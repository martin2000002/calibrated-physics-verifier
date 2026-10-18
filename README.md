# physverify

Calibrated physics verifier for generated video of rigid bodies. Given a clip, `physverify`
extracts object states and contacts (from ground truth or from perception models), evaluates
physics checks R1-R5 (interpenetration, support equilibrium, collision energy, object integrity,
free-fall shape and gravity value) and reports each verdict with the propagated measurement error,
abstaining when the error is too large to decide.

Part of an undergraduate thesis (Computer Science, USFQ). The package map below describes the
architecture.

## Status

Every command is implemented and tested. Check thresholds are provisional until the calibration
activity (A8).

| Command | What it does | Activity |
|---|---|---|
| `dataset` | Kubric synthetic dataset: five scene families, ten perturbation types at four magnitude levels, per-stage ground truth | A4 |
| `extract` | Eight-stage perception pipeline (detection, tracking, depth, camera, 3D lifting, scale with provenance, kinematics, contacts), `fast` (CPU) and `precise` (GPU) profiles | A5 |
| `diagnose` | Extractor error per stage, end to end and by factorial attribution, plus real video (Physics-IQ, PISA) | A6 |
| `data` | Download of the real datasets (Physics-IQ, PISA) | A6 |
| `verify` | Checks R1-R5b (nine measures) on ground-truth ("oracle") or perceived states, with verdict intervals and abstention | A7 |
| `timing` | Per-clip cost harness; only runs on the H200 (`--device cuda --profile precise`) count as measurements | A3 |
| `doctor` | Effective configuration, paths, lock and package versions | - |

Current results come from a local test of the chain on CPU with the `fast` profile and a small set
of clips (32 synthetic, 5 Physics-IQ, 5 PISA). They show that the chain works end to end; they are
not the final experiment. The full dataset, the `precise` profile and the timing measurements run on
the GPU server.

## Installation

Requires Linux, [uv](https://docs.astral.sh/uv/) and, for synthetic data, Docker with the pinned
Kubric image (`runtime.docker_image` in `configs/default.toml`).

```sh
uv sync                                   # everything, with CPU PyTorch
uv sync --no-default-groups --group dev   # light: core + dev tools only
uv sync --no-default-groups --group dev --group cu128 --group perception   # CUDA 12.8 machines
cp configs/local.example.toml configs/local.toml   # machine-local overrides (ignored by git)
```

## Usage

```sh
uv run --no-sync physverify --help
uv run --no-sync physverify doctor
uv run --no-sync physverify --profile precise --device cuda doctor
uv run --no-sync physverify --set runtime.threads=2 doctor
```

Configuration is layered: model defaults, `configs/*.toml`, the active profile, `configs/local.toml`,
`PHYSVERIFY__SECTION__KEY` environment variables, `--device` / `--profile`, then `--set`.
`doctor` prints every effective value with its source.

Exit codes: 0 ok, 1 error, 2 usage, configuration or resource budget, 3 not implemented, 75 heavy
lock busy.

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
| `physverify.schema` | Frozen records shared by host (3.12) and container (3.9): clips, sampling, states, contacts, perception outputs, scale, error profiles, check results; JSON and NPZ codecs |
| `physverify.core` | Plugin discovery, hashing and cache keys, errors and exit codes, logging, resampling, video decoding |
| `physverify.config` | Settings models and the layered TOML loader |
| `physverify.runtime` | Heavy-job lock, Docker runner, thread limits, run directories |
| `physverify.physics` | Geometry (signed distance, contact manifolds, support polygons), finite differences and the numeric definition of every checked quantity |
| `physverify.kubric_worker` | Container entry point and tasks (Kubric, PyBullet, Blender) |
| `physverify.datasets` | Dataset, video and ground-truth protocols and dataset plugins |
| `physverify.synth` | Synthetic scene families and perturbations |
| `physverify.extraction` | Stage providers (perceived and ground truth) and the extraction pipeline |
| `physverify.checks` | Physics checks R1-R5 |
| `physverify.verification` | Check inputs, runner and verdict policies |
| `physverify.diagnosis` | Per-stage metrics, end-to-end error and attribution |
| `physverify.timing` | Timing harness and targets |
| `physverify.cli` | `physverify` command line; subcommands are discovered from `cli/commands/` |

Data, model weights and run outputs live under `data/` (ignored by git).

## Adding a check

A check is one file, `src/physverify/checks/<id>_<name>.py`, that exports a `PLUGINS` tuple. Each class in it
satisfies `physverify.checks._base.Check`: a `check_id`, the `requires` set of inputs it needs, its
`magnitude_unit`, a pydantic `Params` model with its defaults, and `evaluate(inputs)`. Defaults can be overridden
in a `[checks.<id>]` table of `configs/checks.toml`. The registry discovers the file on its own; no other module
changes. Perturbations, stage providers, diagnosis metrics and timing targets follow the same pattern in their
own packages.
