# physverify: rules for agents

The architecture is fixed by ADR 0003, which lives outside this repository at
`../Knowledge/decisions/0003-arquitectura-repo.md` (Spanish). Read it before changing structure.
This file is the English summary that binds every cycle.

## Project rules

- **English only** in this repository: code, identifiers, file names, docstrings, commits, docs.
- **Typed code** (mypy strict on frozen modules), small functions, dependencies injected, no global
  mutable state, configuration only through `configs/**/*.toml` + `physverify.config`.
- **No duplication**: a constant, schema or helper lives in exactly one module. Physics quantities
  are defined once in `physics/{geometry,finite_diff}.py` and `physics/measures/`.
- **Minimal comments**: only formulas, axis conventions and non-obvious decisions.
- **Measurement honesty**: a learned estimate (depth, size, scale) is never reported as an exact
  measurement; it carries its measured error (`sigma`, NaN when still unknown) and is labelled as
  an estimate. Ground truth contributes zero error.
- **Plugins, not edits**: adding a check, perturbation, family, provider, dataset, metric, policy,
  timing target, container task or subcommand means adding one file; no existing file changes.
- Heavy libraries (torch, transformers, kubric, bpy) are imported inside `run()` / functions,
  never at module level; a test enforces that building every registry imports none of them.

## How to run

```sh
uv sync                                   # default groups: dev, cpu (torch +cpu), perception, analysis, data
uv sync --no-default-groups --group dev   # light environment: core + dev only
uv sync --no-default-groups --group dev --group cu128 --group perception   # CUDA 12.8 (H200)
uv run --no-sync physverify --help
uv run --no-sync physverify doctor      # effective config, sources, device, lock, versions
uv run --no-sync pytest                 # unit tests (< 90 s, no network, no GPU, no Docker)
uv run --no-sync pytest -m docker       # Python 3.9 container boundary (needs the Kubric image)
uv run --no-sync ruff check . && uv run --no-sync ruff format --check .
uv run --no-sync ruff check --target-version py39 src/physverify/schema src/physverify/kubric_worker
uv run --no-sync mypy
```

- Use `uv run --no-sync` so uv does not install the default groups behind your back; installs
  are heavy jobs and run under the lock (`flock <lock> uv sync ...`).
- `cpu` and `cu128` are conflicting groups with their own PyTorch indexes; `uv.lock` covers both.
- Global options go before the subcommand: `--config-dir`, `--profile fast|precise`,
  `--device cpu|cuda|cuda:N`, `--set section.key=value` (repeatable), `--no-lock`,
  `--lock-timeout SECONDS`, `--log-level`.
- Config precedence (lowest to highest): model defaults < `configs/**/*.toml` (except the root's
  `local*.toml`) < `[<section>.profiles.<name>]` of the active profile < `configs/local.toml` <
  `PHYSVERIFY__SECTION__KEY` < `--device` / `--profile` < `--set`.
- Exit codes: 0 ok, 1 error, 2 usage / config / resource budget, 75 heavy lock busy.

### Heavy jobs and the lock

The machine runs one heavy job at a time. Commands marked heavy take the flock at
`runtime.heavy_lock` (set in the ignored `configs/local.toml`; see `configs/local.example.toml`).
`physverify doctor` shows the path and the current holder. A parent that already holds the lock
(for example `flock <lock> uv sync ...`, or a scheduler) exports `PHYSVERIFY_LOCK_HELD=1` so
`physverify` does not try to take it again. Never wrap `physverify` itself in `flock`.

## How to add a plugin

Every plugin package is discovered by `physverify.core.registry`: each public module (name not
starting with `_`) is imported and must define `PLUGINS`, a tuple of classes that satisfy the
protocol. Keys must be unique; duplicates, missing `PLUGINS` and protocol mismatches fail loudly.

| Kind | Put one file in | Protocol (in `_base.py`) | Registry key |
|---|---|---|---|
| Physics check | `src/physverify/checks/` | `Check` | `check_id` |
| Perturbation | `src/physverify/synth/perturbations/` | `ScenePerturbation` or `TrajectoryPerturbation` (`target`) | `name` |
| Scene family | `src/physverify/synth/families/` | `SceneFamily` | `name` |
| Stage provider (perceived) | `src/physverify/extraction/providers/` | `StageProvider`, `is_ground_truth = False` | `<stage>/<name>` |
| Stage provider (conditional ground truth) | `src/physverify/extraction/providers/gt_<stage>.py` | `StageProvider`, `name = "ground_truth"`, `is_ground_truth = True` | `<stage>/ground_truth` |
| Dataset | `src/physverify/datasets/` | `DatasetSource` (+ `VideoSource`, `GroundTruthStore`) | `name` |
| Diagnosis metric | `src/physverify/diagnosis/metrics/` | `StageMetric` | `<stage>/<name>` |
| Verdict policy | `src/physverify/verification/policies/` | `VerdictPolicy` | `name` |
| Timing target | `src/physverify/timing/targets/` | `TimingTarget` | `name` |
| Container task | `src/physverify/kubric_worker/tasks/<task>.py` | module with `run(job: dict) -> None`; Python 3.9 syntax | module name |
| CLI subcommand | `src/physverify/cli/commands/<name>.py` | module with `NAME` and a typer `app` | `NAME` |

Notes:

- Providers declare `stage`, `name`, `version`, `requires`, `heavy`, `error_profile_reads` and
  `weights`; their cache key (`core.cache_keys.cache_key`) covers version, params, sampling, upstream
  keys, ground-truth version and only the error-profile entries they declare.
- Ground-truth resampling to a `FrameSampling` happens only in `datasets/_base.py`
  (`ArrayGroundTruthStore`); do not reimplement it.
- Test against `tests/fixtures/synthetic_clip.py` (analytic clip implementing every dataset
  protocol) instead of real data.
- Container code (`schema/**`, `kubric_worker/**`) must run on Python 3.9 and numpy 1.21:
  `from __future__ import annotations`, `typing.Union` for runtime aliases, `class X(str, Enum)`.

## Frozen files

Only the main thread changes these, between phases:

- `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`, `README.md`, `CLAUDE.md`
- `configs/default.toml`, `configs/local.example.toml`, `configs/verifier/size_priors.toml`
- every `__init__.py`, `src/physverify/py.typed`, every `_base.py`
- `src/physverify/schema/**`, `core/**`, `config/**`, `runtime/**`
- `src/physverify/physics/{geometry,finite_diff}.py` and the scaffold's measure modules
  `src/physverify/physics/measures/{penetration,support_overhang,collision_energy,object_integrity,gravity_fit}.py`
  (a check with a new magnitude adds its own module there, owned by its cycle)
- `src/physverify/kubric_worker/__main__.py`, `src/physverify/kubric_worker/tasks/ping.py`
- `src/physverify/cli/main.py`, `cli/context.py`, `cli/commands/doctor.py`
- `tests/conftest.py`, `tests/fixtures/**`, `tests/docker/**`, and the former
  `tests/unit` files: `tests/{core,config,cli,runtime,physics,schema}/**`,
  `tests/kubric_worker/test_kubric_worker.py`, `tests/datasets/test_synthetic_clip.py`,
  `tests/reporting/test_report_text.py`, `tests/test_repo_hygiene.py`

A cycle that needs a contract change does **not** edit a frozen file. It finishes its report with

```
SCHEMA CHANGE REQUEST: <what and why>
```

(or `BUGFIX REQUEST: <what and why>` for `physics/`) and the main thread decides.

## Owners (disjoint files)

| Cycle | Owns |
|---|---|
| A4 dataset | `kubric_worker/build_scene.py`, `kubric_worker/tasks/{simulate,contacts,render}.py`, `synth/*.py`, `synth/families/f*.py`, `synth/perturbations/` except `_base.py`, `datasets/kubric_synthetic.py`, `cli/commands/synth.py`, `configs/datasets/kubric_synthetic.toml`, `tests/synth/**`, `tests/kubric_worker/**` |
| A5 extractor | `extraction/{settings,plan,cache,pipeline,weights,matching}.py`, `extraction/providers/*.py`, `physics/{kinematics,uncertainty}.py`, `datasets/local_video.py`, `cli/commands/extract.py`, `configs/verifier/extraction.toml`, `tests/extraction/**` |
| A7 checks | `checks/r*.py`, `checks/_common.py`, `physics/{equilibrium,tracks}.py`, `verification/{inputs,runner,sweep,scope}.py`, `verification/policies/*.py`, `cli/commands/verify.py`, `configs/verifier/checks.toml`, `tests/checks/**`, `tests/verification/**` |
| A6 diagnosis | `diagnosis/` except `_base.py`, `datasets/{_download,physics_iq,pisa_real}.py`, `annotations/*.json`, `cli/commands/{diagnose,download}.py`, `configs/studies/diagnosis.toml`, `configs/datasets/{physics_iq,physics_iq_views,objectron_static,pisa_real}.toml`, `tests/diagnosis/**` |
| A3 timing | `timing/` except `_base.py`, `cli/commands/timing.py`, `scripts/h200/**` (must not assume Docker), `configs/studies/timing.toml`, `tests/timing/**`, `tests/scripts/**` |
| A8 exclusion rules (WP-J) | `calibration/exclusion_rules.py`, `verification/exclusion_record.py`, `cli/commands/exclusions.py`, `configs/verifier/exclusion_rules.toml`, `tests/calibration/{exclusion_cases,test_exclusions,test_exclusions_cli}.py`, `tests/verification/test_scope_record.py` |
| A8 H1 scope / H3 S5 (WP-J2) | `synth/scene_scope.py`, `configs/studies/primary_scenes.toml`, the H1 primary view in `calibration/{evaluate,artifact,report}.py` and `cli/commands/calibrate.py`, `SECONDARY_TESTS` in `generator_comparison/{preregistered_numbers,prereg,prereg_state}.py`, `tests/calibration/test_monotonicity_primary.py` |
| Campaign infrastructure (WP-O) | `calibration/{dev_score,sealed_ledger}.py`, `configs/verifier/sealed.toml`, `app/sealing.py`, the `dev-score` subcommand in `cli/commands/calibrate.py`, `scripts/h200/run_perception_iteration.sh`, `tests/calibration/{test_devscore,test_sealed,test_sealed_cli,test_cli_devscore}.py`, `tests/scripts/test_run_iteration_script.py` |
| A11 preference / H3 | `finetune/{scorer_scope,stop_rule}.py`, `finetune/scorers/_calibration_status.py`, `cli/commands/{_stop_rule,_finetune_deps}.py`, `tests/finetune/test_{check_scope,stop_rule}.py` |

`configs/verifier/check_scope.toml` is shared by A7 (H2 rule scope) and A11 (H3 scorer scope):
both read it only through `verification/scope.py`.
`configs/verifier/size_priors.toml` belongs to no cycle: it is frozen so that A4 (sampling
synthetic sizes and labels) and A5 (the `scale/size_prior` provider) read the same table.
