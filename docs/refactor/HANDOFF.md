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
| 4b | thin CLI (<=150 lines per command) | todo |
| 5 | split modules >400 lines, layer fixes, `tests/test_architecture.py` | todo |
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
