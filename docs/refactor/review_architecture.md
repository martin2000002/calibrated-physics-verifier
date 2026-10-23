# Architecture review (read-only reviewer, 2026-10-22)

Import graph scripts: scratchpad/graph.py, scc.py.

Verdict: micro-level is good (1 function >60 lines: agreement/pipeline.py:167; 10 uniform registries; config via
Settings.section config/models.py:146). Problem is macro: 14 of 22 subpackages form ONE strongly-connected import
component, and H1/H2/H3 campaign logic is spread over cli, scripts and 7 domain packages.

## 1. Layering
Clean (acyclic): schema, core, config, runtime, physics, checks, kubric_worker.
One 14-package cycle: agreement, calibration, cli, datasets, demo, diagnosis, evaluation, extraction, finetune,
reporting, sizing, synth, timing, verification.
Upward (domain -> cli): diagnosis/checks_port.py:30 (cli.context.AppContext); timing/services.py:12 &
timing/targets/extract_pipeline.py:17 (cli.commands.extract); timing/envinfo.py:21 (cli.commands.doctor);
timing/targets/kubric_render.py:17 (cli.commands.dataset).
Low -> analysis: extraction/plan.py:110-111 (calibration.settings, verification.runner); verification/scope.py:14-15
(calibration); verification/selective_risk.py:24 (diagnosis.pairing); datasets/kubric_fresh.py:15 (calibration.scenes);
datasets/objectron_static.py:48 (sizing); sizing/model.py:16 <-> extraction/providers/scale_size_model.py:27.
Misplaced helpers causing cycles: diagnosis/atomic_io.py (22 importers in 9 pkgs); extraction/weights.sha256_of (12);
calibration/stats.scene_bootstrap used by diagnosis; finetune->timing (Wan2.1 generator in timing/generation.py) while
timing/targets/dpo_step.py -> finetune.
Duplicates: file-sha256 x5 (extraction/weights.py:119, cli/commands/finetune.py:169, calibration/sealed.py:132,
agreement/prereg.py:113, scripts/h200/freeze_query.py:88); atomic write x3 (diagnosis/atomic_io.py:11,
synth/store.py:140, datasets/_download.py:229); bootstrap x3 (calibration/stats.py:98, agreement/stats.py:253,
diagnosis/factorial.py:141); 4 stats modules (calibration, agreement, evaluation, timing).

## 2. Plugin mechanism gaps
- Composition root PLUGIN_KINDS lives in cli/context.py:84-104 -> domain must import cli. checks registry built a
  second time in calibration/settings.py:94.
- Scorers discovered ad hoc (finetune/scoring.py:46), not in PLUGIN_KINDS.
- Judges define PLUGINS (evaluation/judges/*.py) but never discovered; cli/commands/h3.py:42,168-172 hard-wires
  Videophy2AutoevalJudge.
- Adding a check edits synth/perturbations/_measure.py:55-78 (MEASURE_KEYS + dict).
- verification/scope.py:25 time_dependent_checks=("R5b","R4_jump") should be a check attribute.
- Name loops: cli/commands/abstain.py:549,560 & reporting/numbers.py:344 ("raw","calibrated"); reporting/numbers.py:378,
  agreement/report.py:205 ("verifier","vlm").
- Migrations closed list: GT_MIGRATIONS (synth/plan.py), PREDICTIONS/EXTRA_AUDITS (synth/migration_audit.py:966,971).
- Exclusion rules fixed classes calibration/exclusions.py:64-116 (acceptable: preregistered).
- Dead plugin: checks/r4_integrity.py:200 ContinuityCheck (only tests/checks/test_it4_v1_unchanged.py).

## 3. Business logic in CLI
- cli/commands/extract.py is a service locator: 9 modules import pipeline_deps, resolve_dataset, select_clips,
  load_error_profile from it (incl. timing) -> app/services.py.
- cli/commands/_sealed.py, _runs.py, _stop_rule.py, _finetune_deps.py are domain/service code.
- abstain.py: level_tables, collect_synthetic/real, artifact_of, compare_tables, _injected, _z_summary, real_rows ->
  next to verification/selective_risk.py (tests import them directly).
- calibrate.py: Session, _level_data, _real_sources, _fit_meta, _publishable -> calibration service.
- finetune.py: run_record, _check_resume, train_schedule, grid_schedule, require_calibrated, load_pairs,
  campaign_split -> finetune/campaign.py; status_lines -> finetune/status.py; imports private
  cli/commands/timing.py:_timing_directory (finetune.py:35).
- h3.py: H3Context, _current, _require_frozen, _eval_items -> evaluation service.
- exclusions.py: arm_run, replacement, families_without_target -> calibration/exclusions service.
- Target: command = parse options -> one service call -> print, <=150 lines.

## 4. Oversized modules (22 files >400 lines)
synth/migration_audit.py 975 (one-off); cli finetune 817 / calibrate 738 / abstain 570 / h3 559 (§3);
diagnosis/report.py 722 (53 classes) -> report/{models,stage,end_to_end,attribution,real}.py;
calibration/exclusions.py 618 -> rules/evidence/decide; physics/geometry.py 622 (FROZEN) -> poses/sat/contact;
checks/r3_collision_energy.py 585 -> one file per check; verification/sweep.py 539, reporting/numbers.py 508.

## 5. Overlap / one-off
evaluation = H3 only; agreement = H2; calibration = H1 + exclusions + sealed + devscore.
Prereg/freeze in 4 places: agreement/prereg.py, evaluation/prereg.py, calibration/sealed.py, scripts/h200/freeze_query.py.
Abstention in 3: agreement/abstention.py, verification/selective_risk.py, cli/commands/abstain.py.
Report renderers in 7 packages. finetune/budget.py wraps timing/budget.py. sizing = Objectron size-model trainer.
reporting/chain.py reads run_a8v2.sh's directory layout (library depends on a script).
One-off: synth/migrate.py (287), synth/migration_audit.py (975), `dataset migrate|migrate-velocity|audit-migration`
(cli/commands/dataset.py:292-358), scripts/h200/migrate_velocity.sh; ContinuityCheck v1; calibration/heldout_h1.py,
devscore.py.

## 6. scripts/h200
freeze_query.py (570) is real library logic, typed Any, outside mypy, re-implements sha256. a8v2_query.py (224),
h2_query.py (145; imports private _settings from a8v2_query). pv() defined 4x (_common.sh:56, _steps.sh:16,
run_a8v2.sh:233, migrate_velocity.sh:105); run_a8v2.sh copies _steps.sh helpers; die() x2.

## 7. configs
Loader good (flat glob, each table in exactly one file, profiles config/loader.py:57-62). Mapping ad hoc:
diagnosis.toml holds [physics_iq],[pisa_real],[objectron_static],[physics_iq_views]; agreement.toml holds [videophy];
checks.toml holds [verification]; per-experiment files h1_scenes.toml([h1_scope]), h3.toml, kubric_fresh.toml,
abstention.toml, sealed.toml; size_model.toml is a fitted artifact written by `size fit`.

## Proposed target (reviewer)
L0 schema/ core/ (errors, registry, hashing+file_sha256, log, io) stats/
L1 config/ runtime/
L2 physics/
L3 datasets/, synth/, kubric_worker/ (schema only), perception/ (extraction + providers + sizing), generation/
L4 verification/ (checks/, policies/, inputs, runner, scope, sweep, abstention)
L5 studies: calibration (H1), diagnosis, agreement (H2), preference (finetune), generator comparison (H3), timing,
   campaign/ (prereg, sealed ledger, freeze manifest, devscore)
L6 reporting/   L7 app/ (plugins.py composition root, services.py)   L8 cli/ (thin)
Enforce with import-linter layers contract.
Migration order: 1 helpers to core/stats; 2 app/plugins.py; 3 extract helpers -> app/services; 4 hard-coded lists ->
plugin attributes; 5 CLI logic -> services; 6 package renames/regroup (git mv + import rewrite); 7 scripts queries ->
package; shell dedupe; 8 isolate one-off; configs restructure; 9 split god modules.
