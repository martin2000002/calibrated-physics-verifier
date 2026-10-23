# Architecture refactor: handoff

Working notes for the in-progress refactor on branch `refactor/architecture`. This folder is
temporary: it is deleted in the last work package (WP11), before the branch is merged.

## Goal

A behavior-preserving refactor of the whole repo: English everywhere, only necessary comments, no
thesis-process labels, a layered architecture where adding a check, perturbation, model, dataset,
metric, judge, scorer or command means adding one file, and file/folder names that say what they
hold. Everything persisted or hashed stays byte-identical (see `plan.md`, section 0 and the judge
amendments J1-J15).

## Files

- `plan.md`: the approved plan. Precedence: JUDGE AMENDMENTS (J*) > MAIN-THREAD AMENDMENTS (M*) >
  the rest. Paths in it that point to a temporary scratchpad now live here and in
  `tools/equivalence/`.
- `review_architecture.md`, `review_duplication.md`, `review_language_naming.md`: the reviews the
  plan is based on (line numbers refer to base `9eb5888`).
- `tools/equivalence/`: the behavior-equivalence harness (`run_equivalence.sh <base> <head>`,
  `ast_equal.py`); see its README. Rebuild the base output with
  `EQ_EXPECTED= run_equivalence.sh 9eb5888 9eb5888`.
- The decision record, in Spanish and outside this repo, is
  `Knowledge/decisions/0017-refactor-arquitectura.md`. Its "Estado" section logs each WP.

## Process

Each WP goes through four roles. A programmer implements it, with targeted tests only on the laptop.
An auditor grades it against the WP's acceptance criteria. The full suite then runs on the H200,
and the harness must show an empty diff against `9eb5888`. Tags are never moved; `final-freeze` and
the frozen verifier stay untouched. Nothing is pushed without the owner's request.

Full suite on the H200 (CPU only, separate dev clone, never the frozen checkouts):

```sh
git bundle create /tmp/refactor.bundle main refactor/architecture
scp /tmp/refactor.bundle h200:/home/mmonterot/dev/refactor.bundle
ssh h200 'nohup bash /home/mmonterot/dev/wt/refactor_check.sh refactor/architecture <label> >/dev/null 2>&1 &'
# result: /home/mmonterot/dev/wt/logs/refactor-<label>.log (+ .done)
```

`refactor_check.sh` fetches the bundle into `/home/mmonterot/dev/wt/refactor`. It then runs pytest,
`ruff check`, `ruff format --check` and mypy, using `/home/mmonterot/dev/venv` with
`PYTHONPATH=<clone>/src`.

## Status (2026-10-22)

| WP | What | State |
|---|---|---|
| 0 | ADR, `tests/contracts/`, harness | done, audited, H200 green |
| 1 | dead code and one-off scripts | done, audited, H200 green (`0d4b15b`: 3415 passed, 9 skipped) |
| 2 | shared foundations, dedupe | done, audited |
| 3 | composition root `app/`, plugin kinds | done, audit pending |
| 4a | pure package/module renames | done, audit pending |
| 4b | thin CLI (<=150 lines per command) | done (S0, A, B, C merged), fix-ups done, audit pending |
| 5 | split modules >400 lines, layer fixes, `tests/test_architecture.py` | done (S0, A, B, C merged), fix-ups done, audit pending |
| 6 | tests mirror src | todo |
| 7 | config layout `configs/{verifier,datasets,studies}` | todo |
| 8 | CLI, script and env-var renames | todo |
| 9 | English, comments, docstrings, process labels (parallel batches) | todo |
| 10 | stricter mypy and ruff | todo |
| 11 | README, CLAUDE.md, scripts README; delete `docs/refactor/` | todo |

Test counts: the base has 3449 collected; after the WP1 audit fixes, 3424.

The last four WP2 items are done (`0107fcd`..`5142476`): `records_of` comes from
`physics.tracks`, the trainer imports `_torch` from `finetune.dpo`, and the perturbation in-image and
clearance refusals go through `require_in_image`, `require_clear_flight` and `require_clear_shift` in
`_edit.py` with their messages unchanged. The two `write_mp4` stay apart: their encoded bytes match,
but their error types, directory creation, temporary names and failure cleanup differ; only the
fourcc constant is shared. After them, 3616 tests are collected (2 deselected); global ruff and mypy
are clean. The harness from `9eb5888` to `5142476` differs only in the `--help` pages of
`dataset audit-migration`, which WP1 deleted (`2b34c0e`); golden blobs are identical. Next come the
WP2 audit and the H200 run.

Kept separate, with reasons recorded in the ADR:

- `TIE_RTOL`;
- `Interval`, `Ci` and `ProfileInterval`;
- `Coverage` and `CoverageAt`;
- the agreement `CALIBRATED_NAME`;
- the two `NO_DECIDED` values.

`timing/envinfo.py` no longer imports `cli.commands.doctor`: WP3 moved `package_version` to
`runtime/environment.py` (`cdb3193`). `truth_measures` stays for WP5.

Accepted departures from the plan:

- WP1 deleted test-only functions together with their tests, instead of moving them to fixtures.
- WP2 names the run-dir helper `runtime.runs.new_run`, because it returns a `RunContext`.
- The freeze-manifest key paths moved into `tests/contracts/test_freeze_manifest.py`; they are not
  duplicated.
- WP2 benign drifts, persisted bytes unchanged: the store's temporary file name now carries a uuid
  and is removed when the write fails; `run.json` is written atomically; `diagnose` calls `clock()`
  before `config_hash` and `current_commit`.

The WP2 audit PASSED, after its fixes: the last in-image and clearance refusals go through
`_edit.require_in_image` and `_edit.require_clearance`, `RUN_FILE` is `Final`, and the harness
compares against the expected diffs in `tools/equivalence/expected/`.

WP3 (`93b4dcf`..`4952261`) builds every registry through `app.plugins.PLUGIN_KINDS` (11 kinds,
with `scorers` and `judges`) and wires repeated services in `app/services.py`. Departures from the
plan:

- The verification adapter (`A7CheckSuite.build` with explicit dependencies) went first, and the
  scorers commit went before the check-registry commit, because `CalibratedScorer` fell back to
  `check_registry()`.
- The scorer helper is `app.services.context_scorer`, so it does not shadow `scorer` locals; the
  stop-rule and demo helpers take the `AppContext`.
- `app.services.plan_for` and `app.services.chain_runner` wire the plan and the chain runner with
  the context's checks for the four commands that built each one.
- `config.models.validated` holds the model validation that `Settings.section` and the runner
  duplicated; the `[checks]` section lives in `verification/check_set.py`.
- `dataset.py`'s `Run` alias became `runtime.docker.Runner`; `select_clips` gained a one-line
  docstring in `datasets/_base.py`.
- Commands are not a plugin kind; the heavy-import subprocess test discovers them itself, and the
  equivalence driver falls back to `discover_modules` when no kind is `commands`.
- The timing `PlanFactory` takes `TimingServices` to reach the checks.
- `core.registry.protocol_members` counts classmethods and staticmethods, so `ExternalJudge`
  validates; no earlier protocol had one.
- 8 test files patch `plugins.PLUGIN_KINDS`, not 9: the base had 8 files patching it.

WP3 checks: the harness from `9eb5888` to `4952261` exits 0 (only the expected
`wp1-audit-migration.diff`; 59 golden blobs identical); targeted tests, ruff and mypy are clean.

WP4a (`c475333`..HEAD) is the renames of the plan's table, as amended by the judge (JW1, JW6, JW8):
`core.cache_keys`; `calibration.{artifact,artifact_io,dev_score,heldout_monotonicity,exclusion_rules,
sealed_ledger,sigma_recalibration,check_weights}`; `synth.scene_scope` and
`verification.exclusion_record`; `checks.{r2_static_equilibrium,r4_track_jump}`; the seven
`physics.measures` modules; five private provider helpers; seven `diagnosis` modules; the packages
`human_agreement` and `generator_comparison` and their modules; six `finetune` modules;
`timing.target_services`; `sizing.{end_to_end,protobuf_wire,objectron_draws}`; seven `synth` modules.
`LEGACY_PACKAGES` and `_canonical` are gone from `tests/test_architecture.py`. Registry orders are
unchanged (contract tests). Departures:

- Three tool commits (`c475333`, `352ecdd`, `c2c4751`) go before the renames that need them:
  `ast_equal.py --sort-imports` sorts each run of imports (ruff reorders them after a rename), and
  the allowed kinds `alias` and `variable` cover `from physverify.calibration import exclusion_rules
  as exclusion_rules_module` in `test_exclusions.py` (the module now shares its name with the
  `exclusion_rules` function) and the sizing module aliases that share their name with the
  `sizing.e2e` settings attribute.
- Each rename commit `C` passes
  `ast_equal.py . C~1 C --paths src scripts tests tools --sort-imports [--allow docs/refactor/wp4a/<area>.tsv]`
  with exit 0 (allowed lists for calibration, checks, extraction, finetune, timing and sizing), except
  the three commits that edit `tests/test_architecture.py` beyond a rename (`694e17b` drops two
  `KNOWN_VIOLATIONS` entries, `b697070` and `08847b6` empty and delete `LEGACY_PACKAGES`); those
  differ only in that file.
- Comments in `configs/{checks,dataset,exclusion_rules}.toml` that name a renamed module were
  updated; the harness expects them in `expected/wp4a-config-comments.diff`.
- `cli/_sealed` -> `app/sealing`, the check splits, `truth_measures`, the budget merge,
  `timing/generation`, `timing/envinfo`, `timing/mp4`, public provider renames, `selective_risk` and
  `DIFF_PATHS` stay deferred as the plan lists them. Test folders keep their names until WP6
  (`tests/agreement`, `tests/evaluation`, `tests/unit/test_hashing.py`, ...).

WP4a checks: the harness from `9eb5888` to HEAD exits 0 (`wp1-audit-migration.diff` and
`wp4a-config-comments.diff`; 59 golden blobs identical); the full suite passes on the laptop (3620
passed, 2 skipped, 2 deselected); ruff and mypy are clean.

S0 (`3ab707e`..`4bdd55b`, WP4b+WP5 plan S0.1-S0.4) is done: `Settings.section` with PEP 695;
`PipelineDeps.build` shared by `app.services` and `timing.targets.extract_pipeline` (the
`LazyLock`/`RssProbe` plumbing in `extraction/pipeline_resources.py`); `datasets._base.dataset_from`
as the one dataset lookup, with `app.services.{dataset_for,checks_hash_for,scorer_for}` on narrow
`PluginContext`/`LockingContext` protocols; the helpers shared across commands moved below the CLI
(`app/sealing.py`, `app/diagnosis.py`, `runtime.runs.write_json`, `extraction/events.py`,
`datasets._base.fetch_all`, `diagnosis.context.sampled_metrics`, `synth.scene_scope.scene_of`).
Departures:

- `pipeline.py` is 375 lines, not about 345: the `build` classmethod stays in it.
- `PlanRegistries` (the providers/checks pair `perception_plan` needs) lives in
  `timing/targets/extract_pipeline.py`, its only user.
- The private seams named `_source` became `_input_source` (`cli/commands/calibrate.py`,
  `VerifyChecks`) and `_dataset` (`scripts/h200/a8v2_query.py`); tests patch the new names. The
  acceptance grep `def _source` also matches `_sources` and `_source_index`, which are not dataset
  lookups and keep their names (`grep -rn "def _source\b"` is empty).
- `dataset_for` reads `context.settings` before the unknown-dataset check; every caller had read it
  already.
- `guard_sealed` takes `err=`; the CLI passes `cli.context.echo_err` (`typer.echo(..., err=True)`),
  so stderr is unchanged. The repo `CLAUDE.md` still names `cli/commands/_sealed.py`.

S0 checks: the harness from `9eb5888` to `4bdd55b` exits 0 (`wp1-audit-migration.diff` and
`wp4a-config-comments.diff`; 59 golden blobs identical); targeted suites pass (2207 passed, 9
skipped for S0.4); ruff and mypy are clean; the process-label baseline changes only by the
`_sealed.py` -> `sealing.py` move.

## WP4b/WP5

| Part | Scope | State |
|---|---|---|
| S0 | shared seams (`3ab707e`..`4bdd55b`) | done, merged (`bc9b0d2`) |
| C | physics, checks, synth, sizing, diagnosis, extract, scene-stats (`911fde0`..`7863eb5`) | done, merged (`4d22c47`) |
| A | calibration, selective risk, exclusions, verification (`f060f69`..`cbdc58f`) | done, merged (`0aa1bb7`) |
| B | generation, finetune, timing, h3, demo, agreement (`157e378`..`a93454b`) | done, merged (`cbd3f4c`) |
| fix-ups | audit findings of C6-C9 and B (`c947fad`..HEAD) | done, audit pending |

Departures of stream C:

- `write_report` moved from `diagnosis/report.py` to `diagnosis/report_markdown.py` with the
  markdown rendering (the plan kept it in `report.py`); the cell formatters are in
  `diagnosis/markdown_cells.py`.
- `SizePriorParams` is a `FrozenModel` with `title="Params"`, so its JSON schema title is
  unchanged; the `TypeError` text of `expect_params` now names `SizePriorParams`.
- `VIDEO_FILE` moved to `sizing/objectron.py`. `sizing.end_to_end.evaluate_clips` receives the
  scale step as an injected `ScaleOutputs` callable (`app.sizing.scale_outputs`).
- `synth.jobs.JobContext` is a Protocol (settings, registries, docker), so `synth` does not import
  `app`.
- C9 needed extra import edits beyond the moved files (importers of the moved names).
- C9 evidence: `extraction/cache.py:96-110` (`stage_key`) hashes the provider name and version and
  the params dump, not the provider's module path, so the moves do not change cache keys; the
  `doctor` S1 output is equal per the harness (exit 0).

Departures of stream A:

- The exclusion code is four modules (`calibration/exclusion_{rules,decision,inputs,detector_arms}.py`)
  with each rule id named once in `RULE_IDS`; the command wiring is in
  `app/exclusions.py`, and `families_without_target` is in the app layer.
- `specificity` and `null_band` live in `target_ranking`.
- `core/output.Echo` is the line-sink alias; the verify flow is `app/verification.py`.
- A8/J14: `scripts/h200/freeze_query.py` moves to WP8.

Departures of stream B:

- D1: `generation` is a sibling of `verification`
  (`"physverify.verification | physverify.generation"`), not on the synth line as ADR 0017 A.1
  had it, because it imports `extraction` and `datasets`.
- D3: the finetune budget is `timing/finetune_budget.py`, not `runtime/budget` (A.1, JW7), because
  it depends on `timing.records` and `timing.stats`.
- The B5 metrics classes stay in the trainer.
- Extra modules `generator_comparison/analysis_inputs.py` and `reporting/values.py` break an import
  cycle; `app/finetune_training.py` holds the training flows (with `select`).
- `choice()` went to `core/errors` (now `core/choices`, see below).
- `H3Context` holds `Settings` (D7).
- `agreement compare` stayed in the CLI (now moved, see below); the logger name `cli.agreement`
  is kept so log lines do not change.

Fix-ups after the C and B audits (`c947fad`..HEAD):

- `extract run` calls `app.extraction.run_extraction`; it and `app.diagnosis.execute` share
  `app.extraction.recorded_run`. `scene-stats per-clip` wiring is `app.scene_stats.collect_per_clip`.
- `size fetch/fit/eval` call `app.sizing.{fetch_heads,fit_and_write,evaluate_test_split}`.
- The diagnose runners are factories in `app.diagnosis` with the ledger label and `err` injected;
  seams `app.diagnosis.run_stage_errors` and `app.diagnosis.real_datasets`.
- `PluginContext` lives in `config/plugin_context.py`; `JobContext` extends it with `docker`.
  `SIZE_PRIORS_SECTION` lives in `config/models.py`.
- Every `echo`/`err`/`notify` line sink is typed `Echo` and keyword-only in the services touched.
- `agreement compare` calls `human_agreement.workflow.compare_and_report` (seam
  `human_agreement.workflow.write_comparison`); the log timestamp comes from an injected clock.
- `generator_comparison.context.campaign_prompts` and `available_scorer` are gone
  (`finetune.prompts.campaign_prompts`, `app.finetune.active_scorer`); `app.demo.active_scorer` is
  `demo_rule`; `choice()` is in `core/choices.py`; `STOP_RULE_DEVIATION` is `Final` in
  `finetune/stop_rule.py`; `timing.session.report` is the formatter `timing.report.run_lines`.
- `tests/synth/test_truth_measure_pins.py` compares the `float.hex` pins within float tolerance
  (rel 1e-9, abs 1e-12; NaN and `not_applicable` exact): the H200 and the laptop differ in the last
  bits for the same commit.
- No `src` module is over 400 lines and no command module over 150, so nothing else was split.
