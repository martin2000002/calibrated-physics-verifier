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
  `finetune/stop_rule.py`; `timing.session.report` is the formatter `timing.report.run_lines`. Internal order change:
  `run_lines` reads `summary.md` (`timing/report.py:192-193`) before the first line is echoed; the
  old `report` read it after echoing the target and path lines. The output bytes are the same; only
  how much is printed before a failed read differs.
- `tests/synth/test_truth_measure_pins.py` compares the `float.hex` pins within float tolerance
  (rel 1e-9, abs 1e-14, max abs difference measured on the H200 1.45e-15; NaN and `not_applicable` exact): the H200 and the laptop differ in the last
  bits for the same commit.
- Second round, after the audit of `cbd3f4c..6dbf119`: `extract fetch-weights` and `purge` call
  `app.extraction.{fetch_weights,purge_rerendered}`; `scene-stats delta` echoes
  `app.scene_stats.delta_line`; `app.diagnosis.diag_settings` is the one dataset/subset override;
  `guarded_clips` takes `err` keyword-only; `agreement compare` computes its verdict lines before the
  first echo (the old command read `h2_verdict` before any output); `config.models._anchor` is the
  public `anchor` used by `calibration/sealed_ledger.py`.
- No `src` module is over 400 lines and no command module over 150, so nothing else was split.

## WP6

T0 (preparation for the parallel moves T-a and T-b):

- Frozen-file departures, accepted by the WP6 plan (to record in ADR 0017): new
  `tests/fixtures/shell_sandbox.py` (`Sandbox` and `executable`, the base of the eight H200 script
  sandboxes, which keep their specifics in subclasses named after the script) and
  `tests/fixtures/network.py` (the `no_network` autouse guard, one message for all). The guard is
  active only where a conftest imports it: `tests/{timing,human_agreement,scripts,diagnosis,extraction}`. It is not global, so `tests/docker` and Unix sockets are unaffected.
- The harness puts every test helper directory of both layouts on `PYTHONPATH`
  (`HELPER_DIRS` in `run_equivalence.sh`, `CASES_DIRS` in `driver.py prepare`), so `9eb5888` and
  the moved tree run with the same driver.
- `tests/contracts/data/process_labels_baseline.json` is keyed by path: each stream remaps only the
  keys of its moves, and after a merge the baseline is regenerated (`python
  tests/test_process_labels.py`) with totals no higher than before.

T-a (verification, calibration and the split of `tests/unit`):

- `tests/unit` is gone. By main import: `core` (hashing, digests, registry, sampling_video),
  `config`, `cli`, `runtime` (atomic, heavy_lock, runtime, container_engines), `physics`
  (geometry, measures, scale), `schema` (schema_roundtrip), `kubric_worker`, `datasets`
  (synthetic_clip, JY9), `reporting/test_report_text.py`, and `tests/test_repo_hygiene.py` at the
  root. Departure: the plan put sampling_video in `datasets`, but it imports only `core.video`,
  `core.resample` and `schema.clip`, so it goes to `core`.
- `tests/selective_risk/` receives `test_selective_risk.py` and `test_abstain_tables.py` (no
  conftest needed: the latter keeps its own `calibration` insert).
- File renames: `test_h1_by_dataset`/`test_h1_primary` -> `test_monotonicity_by_dataset`/
  `test_monotonicity_primary`, `test_heldout_h1` -> `test_heldout_monotonicity`,
  `d9_golden_case`/`test_d9_golden` -> `dev_score_golden_case`/`test_dev_score_golden` (the golden
  pins the dev-score fit with `sigma_pair_filter="all"`: `golden/dev_score_all_sigma_pairs.json`),
  `checks/test_it4_v1_unchanged` -> `checks/test_r4_jump_v1_reference`,
  `verification/test_it1_oracle_unchanged` + `test_it2_oracle_unchanged` -> `test_oracle_golden`
  (golden `oracle_before_it2.json` -> `oracle.json`), `test_it2_temporal_survival` ->
  `test_temporal_filter_keeps_detections`, `test_it3_contact_survival` ->
  `test_contact_filter_keeps_detections`, `test_it4_camera_inputs` -> `test_camera_inputs`,
  `test_it4_oracle` -> `test_v2_checks_oracle`, `test_it4_targets` -> `test_r4_targets`. Golden
  contents untouched; `r4_v1.json` keeps its name.
- Departures: merging the two oracle files needs one import alias (`oracle_results as
  golden_case_results`, since it1's local `oracle_results` has the same name); the module
  docstrings are joined. One src comment (`extraction/providers/_temporal_smoothing.py`) names the
  renamed test path. Test function names are untouched (JY15: deferred to WP9). The process-labels
  baseline was regenerated: identifier total 285 -> 284 (the `d9` import), the rest equal.

T-b (data, study and script tests, `bd973b0`):

- Moves: `tests/agreement` -> `tests/human_agreement`, `tests/evaluation` ->
  `tests/generator_comparison` (JY9; no finetune test imports `generator_comparison`, so none
  moved), the ten H200 script tests of `tests/timing` -> `tests/scripts`, the extraction goldens
  `test_it3_oracle_unchanged`/`test_it4_floor_refactor` -> `test_oracle_contacts_golden`/
  `test_floor_contacts` and `golden/truth_contacts_before_it3.json` -> `golden/truth_contacts.json`.
  Test functions keep their names (JY15).
- Departure: `test_h2_query.py` and `test_h2_script.py` are in `tests/scripts/human_agreement/`,
  not flat in `tests/scripts/`: their `make_settings` (agreement, `bootstrap_n=50`) and the timing
  one of `test_a8v2_query.py` (data dir in `tmp_path`) share a name, and the autouse
  `provisional_by_default` must reach only them. Their `parents[2]` became `parents[3]`.
- Frozen-file departures (ADR 0017): new `tests/fixtures/videophy_world.py` (the former agreement
  conftest, verbatim) and `tests/fixtures/timing_settings.py` (`data_dir`, `timing_environ`,
  `make_settings`, `make_clip`, `clip_factory` from the timing conftest), because two directories
  use each; the conftests import them as they import `no_network`. `iteration_factory` moved to
  `tests/scripts/conftest.py`, its only user.
- Synth data: `<family>` is the family NAME (`data/clean/f1_projectile`, ...), not the bare
  `projectile` of the plan's D3 example, because `test_truth_measure_pins.py` keys the pins by the
  directory name; `data/r5b` -> `data/gravity_scale`. Path constants updated in
  `tests/synth/{conftest,test_contacts_gt,test_gt_export,test_migrate,test_truth_measure_pins}.py`
  and `tests/kubric_worker/test_render_docker.py` (outside T-b's folders, one line). Nothing in
  `src/`, `tools/` or `scripts/` read these paths.
- Repo `CLAUDE.md` names `tests/scripts/test_run_iteration_script.py`.
- Checks: 3628 collected / 2 deselected before and after, sorted test ids (without path) equal,
  per-test fixture sets of `--setup-plan` equal (`no_network` on 1416 tests before and after);
  1552 passed over the moved directories; process-label totals unchanged (424 keys).

## WP7

Configuration layout (`refactor/wp7-k`, from `5c62128`; K4 has no commit of its own: the test
consumers had to move with the files in K2 to keep the suite green).

- Layout: only `default.toml` and `local.example.toml` stay in `configs/`. `verifier/`: checks,
  check_scope, exclusion_rules, extraction, calibration, sealed, sizing, size_priors, size_model.
  `datasets/`: `kubric_synthetic` (was `dataset.toml`), kubric_fresh, physics_iq,
  physics_iq_views, objectron_static, pisa_real, videophy. `studies/`: diagnosis,
  `primary_scenes` (was `h1_scenes`), `human_agreement` (was `agreement`),
  `generator_comparison` (was `h3`), `selective_risk` (was `abstention`), finetune, timing, demo.
  `diagnosis.toml` was split into `studies/diagnosis.toml` and the five dataset files, and
  `agreement.toml` into `studies/human_agreement.toml` and `datasets/videophy.toml`; the blocks
  moved verbatim, comments included (the concatenation of the pieces equals the old file modulo
  blank separator lines; the agreement header now precedes [agreement]; pinned by
  `tests/contracts/test_config_splits.py` against copies of the old files in
  `tests/contracts/data/pre_split_configs.json`). Section
  and key names are unchanged, so `config_hash` is too (`test_config_hashes.py` unchanged and
  green; `doctor` prints the same `config_hash:` for both profiles).
- Loader (`config/loader.py`, frozen-file departure): `configs/**/*.toml`, sorted by path, with
  `local*.toml` skipped only at the root (a `local.toml` in a subdirectory is read). `sources`
  stay absolute; only the duplicate-table error names the files relative to `configs/` (JY14,
  departure from plan.md C). `config_files` is public so the test fixture reuses it.
  `cli/main.py` is unchanged (its `configs/` default is still right).
- `app/sizing.py`: `MODEL_FILE = "verifier/size_model.toml"`; `size fit` creates `verifier/`
  through `core.atomic.write_atomic` (tested in `tests/sizing/test_fit_cli.py`).
- Frozen-file departures (ADR 0017): `config/loader.py`; `configs/size_priors.toml` moved to
  `configs/verifier/`; `tests/conftest.py` and `tests/synth/conftest.py` copy the configs through
  the new `tests/fixtures/repo_configs.py` (`copy_repo_configs`, subdirectories kept);
  `README.md`, `scripts/h200/README.md` and `CLAUDE.md` name the new paths.
- Scripts: `run_final.sh` `v1_configs` (JY1) flattens the freeze's tree for the v1 checkout (which
  reads only `configs/*.toml`), stops on two files sharing a basename, overlays the v1 checkout's
  `checks.toml`, compares file by file (only `checks.toml` may differ; a file on one side only
  stops) and still checks a rerun against `configs.diff`. A v1 arm directory written before WP7
  cannot be rerun: its `configs.diff` names the old flat paths. `setup_autoeval.sh` reads the pins
  from `configs/studies/generator_comparison.toml` (J5); `run_h2.sh` names
  `configs/studies/human_agreement.toml`.
- Kept verbatim and now stale (follow-up): comments inside the configs that name old paths
  (`studies/finetune.toml` line 2, the check_scope note in `studies/human_agreement.toml`,
  `verifier/checks.toml` line 7, `verifier/sizing.toml` line 2).
- Harness: expected diff `tools/equivalence/expected/wp7-config-layout.diff`;
  `wp4a-config-comments.diff` retired; `wp1-audit-migration.diff` realigned (same lines). `9eb5888`
  vs WP7: exit 0, no `config_hash` or JSON difference, 59 golden blobs identical.

## WP8

Stream S (CLI and script renames, J14), branch `refactor/wp8-s`:

- CLI renames: `h3` -> `compare-generators`, `abstain` -> `selective-risk`, `data` -> `synth`,
  `dataset` -> `download`, `calibrate devscore` -> `calibrate dev-score`; `results` options
  `--a8v2-dir`/`--h1-evaluation`/`--h2-run`/`--abstention-run` -> `--chain-dir`/
  `--calibration-evaluation`/`--agreement-run`/`--selective-risk-run` (the `Request` fields keep
  their names). Persisted names (JY6) are untouched: sealed-ledger labels
  (`app/sealing.LEDGER_LABELS_BY_LEAF` maps the renamed leaves onto the old labels), step names
  in `steps.tsv`, `state.env` keys, `PHYSVERIFY__*` in `ENV_SHA`, `devscore.json`/`DEVSCORE_SHA`,
  the freeze schema, `Candidate` fields, `--p2-commit`, and `ABSTAIN_CURVE` in
  `reporting/chain.py` (the log name of the retired chain runner).
- Script renames: `run_h2.sh` -> `run_human_agreement.sh`, `h2_query.py` -> `agreement_query.py`,
  `run_h3.sh` -> `run_generator_comparison.sh`, `run_iteration.sh` ->
  `run_perception_iteration.sh`, `iteration_arms.tsv` -> `perception_arms.tsv`, `a8v2_query.py`
  -> `run_query.py`, `setup.sh` -> `setup_cuda.sh`. Variables: `PHYSVERIFY_H2_A8V2_DIR` ->
  `PHYSVERIFY_AGREEMENT_CHAIN_DIR`, `PHYSVERIFY_H2_*` -> `PHYSVERIFY_AGREEMENT_*`,
  `PHYSVERIFY_ITER_*` -> `PHYSVERIFY_ITERATION_*`, `PHYSVERIFY_FREEZE_P2_COMMIT` ->
  `PHYSVERIFY_FREEZE_EARLY_EXTRACTION_COMMIT`. `_common.sh` holds the only `pv` (`PV_RUNNER`,
  `PV_PROFILE`, `PV_NICE`, `PV_EXTRA`) and `h200_py`; `run_query.py` exports `query_settings`
  (was `_settings`) for `agreement_query.py`.
- J14: the freeze queries live in `physverify.freeze` (`errors`, `files`, `selection`, `diffs`,
  `manifest`, `fresh_render`, `ledger`); `scripts/h200/freeze_query.py` keeps the settings, the
  parser, the dispatch and `main`. The package sits at the `finetune` layer, so it does not
  import `physverify.app`: the script hands `record_sha` a callable that builds the check
  registry, called inside its `PhysverifyError` -> `QueryError` guard and only after the J3 and
  v1 refusals (`manifest` takes `record` and `now`). The script imports only `errors`, `files`,
  `selection` and `diffs` at the top; the settings, plugins, manifest, render and ledger modules
  are imported inside the commands, so `field` runs in about 0.07 s (3.4 to 3.8 s with the eager
  imports). `tests/contracts/test_freeze_manifest.py` pins the bytes of
  a fixture freeze.json, computed with the pre-move script.
- Frozen-file departures (ADR 0017): `README.md` (command table), `CLAUDE.md` (owner rows:
  `cli/commands/synth.py`, `{diagnose,download}.py`, the `dev-score` subcommand,
  `run_perception_iteration.sh`), `pyproject.toml` (mypy `files` += `scripts/h200`, `mypy_path`
  `src:scripts/h200`, importlinter layer `physverify.finetune | physverify.freeze`), and the new
  `src/physverify/freeze/__init__.py`.
- Other departures: `tests/contracts/test_registries.py` renames
  `test_command_names_and_order_are_unchanged_until_the_cli_renames` to
  `test_command_names_and_order_are_pinned` (its name stated the WP8 deadline; an exception to
  JY15). `tests/reporting/{test_chain,reporting_cases}.py` build the historical log lines from
  `ledger_label(...)` and `ABSTAIN_CURVE`. `physverify.freeze.files` and `.diffs` are modules
  the plan did not list (shared readers, and the J3/J13 diff paths the script's parser needs
  without importing `manifest`).
  The fixture freeze uses the iteration id `it6-chosen`, so the pin adds no process label.
- Left to stream K: the `configs/sealed.toml` comment that still names `run_h2.sh`.
- Equivalence: `tools/equivalence/expected/wp8-cli-renames.diff` declares the help pages, the
  stderr hints and the missing-input lines of `results_chain` and
  `results_chain_without_agreement`. The renamed help pages shift the alignment of
  `diff -r`, so the audit-migration hunk header in `wp1-audit-migration.diff` is now
  `318,334d314` (was `316,332d312`); its lines are otherwise unchanged.
