# physverify: rules for contributors and agents

Calibrated physics verifier for generated video. `README.md` describes what it does and how to
run it; this file says how the code is organised and what must not change.

## Rules

- **English only** in this repository: code, identifiers, file names, docstrings, commits, docs.
- **Typed code** (mypy strict on `src/physverify`, `scripts/h200` and `tools/equivalence`), small
  functions, dependencies injected, no global mutable state, configuration only through
  `configs/**/*.toml` and `physverify.config`.
- **No duplication**: a constant, schema or helper lives in exactly one module. A physical quantity
  that both a check and its truth measure need is computed once, in `physics/measures/`.
- **Minimal comments**: only formulas, axis conventions and non-obvious decisions. No banners, no
  plan or study labels (`tests/test_process_labels.py` enforces this).
- **Measurement honesty**: a learned estimate (depth, size, scale) is never reported as an exact
  measurement; it carries its measured error (`sigma`, NaN when still unknown) and is labelled as
  an estimate. Ground truth contributes zero error.
- **Plugins, not edits**: adding a check, perturbation, model, dataset, metric or command means
  adding a file (see the table below), not changing existing code.
- **Heavy libraries** (torch, transformers, kubric, bpy) are imported inside functions, never at
  module level; `tests/core/test_registry.py` checks that building every registry imports none.
- **Size limits**: no module in `src/physverify` over 400 lines, no CLI command module over 150
  (`tests/test_module_sizes.py`). Commands only parse arguments and call `physverify.app`.

## Package tree

```
src/physverify/
  schema/          wire records shared with the Python 3.9 container; JSON and NPZ codecs
  core/            plugin discovery, errors and exit codes, hashing and cache keys, atomic writes,
                   downloads, markdown tables, video decoding, resampling
  stats/           bootstrap, scene bootstrap, folds, intervals, rank statistics
  config/          settings models, layered TOML loader, plugin context protocol
  runtime/         heavy-job lock, Docker and udocker runners, run directories, environment info
  physics/         geometry, finite differences, kinematics, equilibrium, measures/
  synth/           Kubric scene families/, perturbations/, truth_measures/, ground-truth export
  sizing/          object size model learned on Objectron
  datasets/        dataset, video and ground-truth protocols and one module per dataset
  checks/          physics checks R1-R5b, one module per check
  extraction/      stage providers (providers/), stage cache, extraction pipeline
  verification/    check set, runner, verdict policies/, specificity sweep, check scope
  generation/      Wan2.1 text-to-video generator
  diagnosis/       per-stage metrics/, end-to-end error, attribution, real video
  calibration/     thresholds, rates, detection limits, sigma recalibration, clip score,
                   held-out evaluation, exclusion rules, sealed-set ledger
  human_agreement/ verifier vs reference VLM on human labels
  selective_risk/  risk-coverage curve and abstention level
  finetune/        preference pairs, scorers/, Flow-DPO trainer, checkpoint selection
  freeze/          freeze manifest and the queries of scripts/h200/freeze_query.py
  generator_comparison/  fine-tuned vs base vs best-of-8, judges/
  timing/          per-clip cost harness, targets/, GPU budget
  demo/            one prompt or video through the whole verifier
  reporting/       results report: every number with its source
  app/             composition root: plugins.py (PLUGIN_KINDS), context.py, services
  cli/             typer entry point; commands/ holds one module per subcommand
  kubric_worker/   container entry point and tasks/ (Python 3.9)
tests/             mirrors src/ (tests/<package>/), plus scripts/, contracts/, equivalence/, fixtures/
scripts/h200/      shell runners and small query scripts for the university H200 server
tools/equivalence/ behaviour-equivalence harness (see its README)
configs/           default.toml, local.example.toml, verifier/, datasets/, studies/
```

## Layer rule

`[tool.importlinter]` in `pyproject.toml` lists the layers; `tests/test_architecture.py` enforces
them with `ast`, lazy imports inside functions included. A package imports only from the layers
below it, never from a sibling (`a | b` on one line) or from above. Top to bottom:

```
cli > app > reporting > generator_comparison | timing | demo > finetune | freeze
  > human_agreement | selective_risk > calibration > diagnosis > verification | generation
  > checks | extraction > datasets > synth | sizing > physics > runtime > config > stats
  > core > schema
```

`schema` imports nothing else from `physverify`; `kubric_worker` imports only `schema`; neither
imports pydantic, typer or torch. `scripts/` may import `physverify.app` and below, never
`physverify.cli`; ruff (TID251) also bans `physverify.cli` and `physverify.app` outside `cli`,
`app`, `scripts`, `tools` and `tests`.

## How to add a plugin

`app/plugins.py` discovers every kind through `core.registry.discover`: each public module (name
not starting with `_`) of the package is imported and must define `PLUGINS`, a tuple of classes
that satisfy the kind's protocol. Duplicate keys, a missing `PLUGINS` and protocol mismatches fail
loudly. Modules load in file-name order and the registry keeps that order, which is the output
order of reports and tables.

| Kind | Add one file | Interface it implements | Registry key |
|---|---|---|---|
| Physics check | `checks/r<id>_<what>.py` (file order = check order) | `checks._base.Check` | `check_id` |
| Truth measure of a check | `synth/truth_measures/r<id>_<what>.py` | `synth.truth_measures._base.TruthMeasure` | `check_id` |
| Perturbation | `synth/perturbations/<name>.py` | `ScenePerturbation` or `TrajectoryPerturbation` (`synth.perturbations._base`) | `name` |
| Scene family | `synth/families/<name>.py` | `synth.families._base.SceneFamily` | `name` |
| Perception model | `extraction/providers/<stage>_<name>.py` | `extraction._base.StageProvider`, `is_ground_truth = False` | `<stage>/<name>` |
| Ground-truth stage | `extraction/providers/gt_<stage>.py` | `StageProvider`, `name = "ground_truth"`, `is_ground_truth = True` | `<stage>/ground_truth` |
| Dataset | `datasets/<name>.py` (and `configs/datasets/<name>.toml` if it has settings) | `datasets._base.DatasetSource` (with `VideoSource`, `GroundTruthStore`) | `name` |
| Diagnosis metric | `diagnosis/metrics/<stage>_<metric>.py` | `diagnosis._base.StageMetric` | `<stage>/<name>` |
| Verdict policy | `verification/policies/<name>.py` | `verification._base.VerdictPolicy` | `name` |
| Timing target | `timing/targets/<name>.py` | `timing._base.TimingTarget` | `name` |
| Comparison judge | `generator_comparison/judges/<name>.py` | `generator_comparison.judges._base.Judge` (or `ExternalJudge`) | `name` |
| Preference scorer | `finetune/scorers/<name>.py` | `finetune.scorers._base.AggregateScorer` | `name` |
| Container task | `kubric_worker/tasks/<task>.py` (Python 3.9 syntax) | a module with `run(job: dict[str, Any]) -> None` | module name |
| CLI subcommand | `cli/commands/<name>.py` | a module with `NAME` and a typer `app` (discovered by `cli/main.py`) | `NAME` |

Paths are under `src/physverify/`. Every check has exactly one truth measure, so a check graded
against synthetic ground truth adds two files. A new subcommand is also listed in
`app/sealing.py`, as `GUARDED` (it computes on data and must respect the sealed sets) or in
`EXEMPT` with its reason; `tests/calibration/test_sealed_cli.py` checks that every subcommand is
one or the other. The contract tests pin registry keys and orders, so a deliberate new plugin
adds its key there in the same commit.

Notes:

- Providers declare `stage`, `name`, `version`, `requires`, `heavy`, `error_profile_reads` and
  `weights`; their cache key (`core.cache_keys.cache_key`) covers version, params, sampling,
  upstream keys, ground-truth version and only the error-profile entries they declare.
- Ground-truth resampling to a `FrameSampling` happens only in `datasets/_base.py`
  (`ArrayGroundTruthStore`); do not reimplement it.
- A check's `Params` defaults can be overridden in a `[checks.<id>]` table of
  `configs/verifier/checks.toml`.
- Test against `tests/fixtures/synthetic_clip.py` (an analytic clip implementing every dataset
  protocol) instead of real data.
- Container code (`schema/**`, `kubric_worker/**`) must run on Python 3.9 and numpy 1.21:
  `from __future__ import annotations`, `typing.Union` for runtime aliases, `class X(str, Enum)`.

## Contracts

These values are written to disk, hashed or parsed by the scripts. Stored runs, caches and
artifacts stay readable only while they are unchanged, so a change here is a format change, never
a cleanup. `tests/contracts/` pins each one.

- Config table and key names and values: they enter `Settings.config_hash()`,
  `look_config_hash`, `checks_hash`, `DatasetSettings.content_hash()` and every provider's
  `stage_key` (`test_config_hashes.py`, `data/config_key_paths.txt`). Files may move between
  folders of `configs/`; tables and keys may not be renamed.
- Registry keys and order: check ids, perturbation and family names, dataset names and subsets,
  provider keys, metric and policy names, command names (`test_registries.py`).
- Field names and order of every persisted model and of each check and provider `Params`
  (`test_persisted_fields.py`, `data/persisted_fields.json`), and the `DatasetPlan` JSON.
- Class names of `Record` types (`schema.io.envelope` writes `type(record).__name__`) and of
  `PhysverifyError` subclasses (written by diagnosis and extraction failures); schema ids and
  versions (`*_SCHEMA`, `*_VERSION`), `GT_REVISION`, persisted file names and the `run.json`
  fields (`test_persisted_names.py`).
- Sealed-ledger command labels (`app/sealing.py`: `GUARDED`, `LEDGER_LABELS_BY_LEAF`): a renamed
  subcommand keeps its old label in the ledger.
- Output lines that `scripts/h200` parses with `sed`, `grep` or `awk`, and the `doctor` keys the
  runners read (`test_parsed_output.py`).
- The bytes, schema and key paths of `freeze.json` (`test_freeze_manifest.py`).
- Serialized names that keep a study label (`h1_pass`, `h2_verdict`, `[h3.autoeval]`, ...): they
  stay as they are, without aliases; `tests/test_process_labels.py` lists each with its reader.

## How to run

```sh
uv sync                                   # default groups: dev, cpu (torch +cpu), perception, analysis, data
uv sync --no-default-groups --group dev   # light environment: core + dev only
uv run --no-sync physverify --help
uv run --no-sync physverify doctor        # effective config, sources, device, lock, versions
uv run --no-sync pytest                   # no network, GPU or Docker
uv run --no-sync pytest -m docker         # Python 3.9 container boundary (needs the Kubric image)
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

### Behaviour equivalence

`tools/equivalence/run_equivalence.sh <base> <head>` runs the same CPU scenarios on two revisions
and diffs their outputs; intended differences are declared in `tools/equivalence/expected/`. Use
it for any change meant to keep behaviour.
