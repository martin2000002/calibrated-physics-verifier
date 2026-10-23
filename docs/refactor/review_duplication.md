# Duplication / dead code / practices review (read-only reviewer, 2026-10-22)
Scripts: scratchpad/ana.py, consts.py, dead.py, show.py. Paths relative to src/physverify/.

Overall disciplined: 3660 functions, none untyped, no bare except, 1 env read (cli/main.py:119), no mutable globals.

## Duplicated logic
- Run-dir creation (9-line RunContext.create block) x7: shared cli/commands/_runs.py:16 new_run_dir; copies in
  verify.py:152 _run_dir (identical), exclusions.py:269 _run, timing.py:225 _new_run, calibrate.py:147, extract.py:111,
  diagnose.py:99.
- Atomic writers x5: diagnosis/atomic_io.py:11 write_atomic (best); synth/store.py:140 atomic_write_bytes, :156
  atomic_save (pid temp, no cleanup); datasets/_download.py:229 save_atomic; kubric_worker/build_scene.py:316-335
  (py3.9 isolation, justified); runtime/runs.py:75 writes run.json non-atomically.
- write_mp4 x2: core/video.py:35 (MP4_FOURCC) and timing/mp4.py:39 (FOURCC); finetune/generate.py:34 imports timing's.
- Markdown _table x6: calibration/report.py:69, diagnosis/report.py:370, agreement/report.py:116,
  evaluation/report.py:59, timing/report.py:253, reporting/render.py:80.
- Number/cell formatters ~10 with divergent missing markers: cli/commands/calibrate.py:737 _num, demo/report.py:203,
  evaluation/report.py:53, cli/commands/verify.py:209 _number, diagnose.py:239, size.py:175, timing/report.py:244,
  agreement/report.py:72 _fmt, diagnosis/report.py:356 _cell.  (CAUTION: outputs are report text; changing the
  missing marker changes report bytes -> keep each call site's marker via a parameter.)
- sha256: file hashers extraction/weights.py:119 sha256_of(path), cli/commands/finetune.py:169 _sha256,
  synth/gt_export.py:327 content_sha256, scripts/h200/freeze_query.py:88; text hasher calibration/loader.py:68
  sha256_of(text) (same name, different meaning).
- ConfigDict(frozen=True, extra="forbid") redefined 38x in 26 files though config/models.py:45 FrozenModel exists.
- _checks_hash identical cli/commands/calibrate.py:174 & h3.py:164; parse_split finetune.py:144 == h3.py:367 _split;
  records_of physics/tracks.py:145 == synth/perturbations/_windows.py:43; _torch() lazy import x3 (finetune/dpo.py:20,
  finetune/trainer.py:44, timing/envinfo.py:55); perturbation "centres in image + clearance" _check x5
  (r1_penetration.py:123, r3_energy_gain.py:147, r3_restitution_gain.py:113, r5a_gravity_switch.py:175,
  r5a_object_gravity.py:115).
- Interval models: diagnosis/report.py:133 Interval vs calibration/stats.py:221 ProfileInterval (+ calibration/stats.Ci);
  Coverage diagnosis/report.py:56 vs calibration/schema.py:157 CoverageAt. (CAUTION: serialized field names.)
- generic bootstrap in diagnosis/factorial.py:141 imported by calibration, agreement, evaluation, sizing.
- scripts: a8v2_query.py:139 == freeze_query.py:439 _json; overlapping _settings; outside mypy.
- Constants: RUN_FILE "run.json" x4 + literal (agreement/run_io.py:26, cli/commands/exclusions.py:80,
  cli/commands/timing.py:74, finetune/store.py:58, runtime/runs.py:75); BYTES_PER_MB x3 (cli/commands/data.py:26,
  extract.py:41, datasets/objectron_static.py:61); NO_DECIDED "no_decided_check" (calibration/apply.py:22,
  finetune/scorers/provisional.py:32) vs divergent "no_decided" finetune/pairs.py:30 (CAUTION: may be a persisted
  value -> verify before unifying); SIGMA_LEVELS (cli/commands/finetune.py:118, timing/targets/dpo_step.py:34);
  CALIBRATED/PROVISIONAL (demo/report.py:26-27, verification/runner.py:34-35, agreement/decision.py:27); TIE_RTOL
  (checks/_common.py:33, extraction/providers/_impacts.py:45); SWITCH_DELAY_S (both r5a perturbations);
  "local_video" x4; "video.MOV" x2; hash truncation widths 8/10/12 in 7 constants.

## Dead code
- Unreferenced: cli/context.py:184 not_implemented + core/errors.py:32 NotImplementedCommandError + exit code 3 +
  OWNER/IMPLEMENTED plugin fields (all 17 commands IMPLEMENTED=True); calibration/collect.py:49 run_level;
  synth/perturbations/_windows.py:241 centre_in_image, :245 all_ids; runtime/threads.py (thread_env test-only;
  runtime/docker.py:34 rebuilds same env vars).
- Test-only (11): physics/finite_diff.py:66,:71; synth/layout.py:101; timing/stats.py:112; calibration/apply.py:75;
  calibration/loader.py:272; evaluation/state.py:212; diagnosis/metrics/lift_gravity.py:33;
  synth/perturbations/_ballistics.py:183; datasets/_videophy_tables.py:206; checks/r4_integrity.py:200 ContinuityCheck.
- One-off: synth/hdri.py (python -m printer, 3 of 6 print() calls); synth/migration_audit.py (975), synth/migrate.py
  (WP-G/H/I migration audits; live only via `dataset migrate` / `migrate-velocity`).

## Typing
0 missing annotations, 9 type: ignore (3 in schema/scene.py:69,70,196); Any 687 (worst: synth/migration_audit.py 41,
schema/io.py 39, cli/commands/abstain.py 35, finetune/trainer.py 26, agreement/report.py 22,
kubric_worker/build_scene.py 22); 324 dict[str, Any] payloads. mypy strict relaxed for 9 packages (61% of lines) —
unnecessary now. Ruff lacks ANN, BLE, T20, PL, RUF.

## Size
Only agreement/pipeline.py:167 compare >60 lines (69). Next: evaluation/analysis.py:390 (59), evaluation/compare.py:158
(58), synth/perturbations/r3_energy_gain.py:56 (54), cli/commands/finetune.py:671 (53). Nesting max 5
calibration/exclusions.py:395. Real problem: fat command modules.

## Errors
0 bare except; 15 except Exception all re-raise/record. print() only cli/main.py:110,113, kubric_worker/__main__.py:57,
synth/hdri.py.

## Tests
10 iteration-named files; 174 test functions named after plan items (test_t14_...) in 41 files; 24 fixtures duplicated
(class Sandbox x11 in script tests; no_network/_refuse x4 conftests; make_settings x6; settings x14); ~5.1k lines of
shell-script orchestration tests (test_run_a8v2_script 869, test_run_final_script 765, test_h2_script 608,
test_run_iteration_script 470); 9 private imports.
