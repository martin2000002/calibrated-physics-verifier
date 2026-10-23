# Master plan: behavior-preserving architecture refactor of physverify (planner v1 + main-thread amendments)

Base main @ 9eb5888 (clean). Branch `refactor/architecture`. Verifier frozen at tag `final-freeze` (0641659); never
move/delete tags, never push. Baseline suite (laptop): 3446 passed, 3 skipped, 2 deselected (~10:40). ruff, ruff
format, mypy clean. R = /home/martin/Desktop/Main/USFQ/Semestre_9/Poryecto Integrador/Repo.

## MAIN-THREAD AMENDMENTS (override the planner text below where they conflict)
M1. ADR number is **0017** (`Knowledge/decisions/0017-refactor-arquitectura.md`); 0016 is reserved for the "ronda v2"
    ADR drafted in Knowledge/handoffs/mejora-percepcion/plan-it5.md.
M2. O1 modified: KEEP the generic migration mechanism (`synth/migrate.py`, `synth/plan.GT_MIGRATIONS`,
    `GT_REVISION`, the `migrate` and `migrate-velocity` subcommands, `StageCache.purge_clip`/`extract purge` if used by
    migrate flows): a later round (it5: `occluded_at` changes the oracle) may need a GT revision bump on stored
    datasets. Make migrations scalable if cheap (one module per migration or keep the ordered tuple; decide with
    evidence). DELETE the one-off audits: `synth/migration_audit.py`, `audit-migration` subcommand, their tests (move
    any live-physics assertions to fixtures), `scripts/h200/migrate_velocity.sh`, `scripts/h200/run_a8v2.sh` and
    the parts of `a8v2_query.py` only they use. `reporting/chain.py` keeps reading recorded chain dirs.
O2 accepted (delete run_p1_perception.sh + test). O3: NO new dependency (no lock change): the AST architecture test
is the gate. O4 accepted (persisted report text keeps its labels; allow-list). O5: env renames listed in ADR; the main
thread updates server launchers.
M3. Tests run on the remote server H200 via the main thread (programmers run only targeted subsets locally, the
    laptop has 4 threads/7.6 GB; never the full suite locally unless told). Programmers commit on the branch; the main
    thread ships the branch to the server and runs the full suite + ruff + mypy there.
M4. it5 (a later research round) will build on this branch: run_iteration.sh, run_final.sh, run_freeze.sh,
    run_h2.sh, run_h3.sh, run_kubric.sh, run_preference.sh, run_timing.sh, run_demo.sh, run_finetune.sh, setup*.sh
    must keep working (renamed is fine, tests/scripts green).

## 0. Findings that constrain the plan (verified in code by planner)
1. Config TABLE/KEY names are wire data: `Settings.config_hash()` (config/models.py:153) hashes all raw sections ->
   run.json + report headers; `look_config_hash` (calibration/sealed.py) -> sealed ledger;
   `DatasetSettings.content_hash()` checked by `DatasetPlan.merged`; `checks_hash` dumps check Params field names;
   `stage_key` -> core/hashing.cache_key covers provider name/version + params.model_dump. => section & key names
   stay ([h1_scope], [h3*], h2_dev, time_dependent_checks…), `PHYSVERIFY__SECTION__KEY` env vars stay, check/provider/
   DatasetSettings/GenerationSettings field names stay. Module paths are never hashed -> modules move freely. File
   names and which file holds which section are free (loader merges top-level tables, one file per table).
2. Registry order is output order: `discover` imports modules sorted by FILE NAME, then PLUGINS tuple order;
   `build_checks` iterates the registry -> check order R1,R2,R3a,R3b,R4_missing,R4_size,R4_jump,R5a,R5b is
   checks.jsonl/report order. `plan_subset` sorts families/perturbations by NAME (file names irrelevant).
3. schema Record CLASS NAMES are serialized (`schema/io.envelope` writes type(record).__name__) -> no schema class
   renamed.
4. Persisted schema-id strings: "physverify.devscore", "physverify.calibration", "physverify.calibration.evaluation",
   "physverify.results", FREEZE_SCHEMA keep values.
5. Rendered reports contain process labels ("## H1 primary", "decides H2", "# H3 comparison", "ADR 0011 D6",
   "provisional until the A8 threshold calibration", "h1_pass primary") -> literal allow-list; stdout/stderr-only
   text is cleaned unless a script parses it (doctor lines in run_freeze.sh, "abstain curve" in reporting/chain.py).
6. Serialized jargon fields stay as attributes (no aliases): h1_pass, meets_h1, heldout_h1, h1_primary, h2_verdict,
   h2_group, h3_families_without_target, h3_report.json/.md, freeze.json diffs.j3/j13/v1/record_sha. NO_DECIDED has
   two persisted values ("no_decided" finetune/pairs; "no_decided_check" calibration/apply + finetune/scorers): do not
   merge; rename constants after where written.
7. v1 ContinuityCheck (checks/r4_integrity.py:200) not in registry; final run's v1 arm runs from a separate C_V1
   checkout; used only by tests (oracle_golden_case.py V1_OVERRIDES for golden/oracle_before_it2.json, test_r4.py,
   test_sigma.py, test_it2_temporal_survival.py, test_it3_contact_survival.py, r4_v1_golden_case.py).
8. Migrations needed only by run_a8v2.sh step 0 (see M2 for the decision).
9. Upward imports into cli: diagnosis/checks_port.py:30, timing/services.py:12, timing/targets/extract_pipeline.py:17,
   timing/targets/kubric_render.py:17, timing/envinfo.py:21, scripts/h200/freeze_query.py:206,377.
10. import-linter not installed -> stdlib AST test reading a contract from pyproject.

## A. Target architecture
### A.1 Layers (bottom -> top)
0 schema (py3.9 wire records; stdlib/numpy only)
1 core: errors, registry (discovery only), cache_keys (was hashing), digests (file/text sha256), atomic (from
  diagnosis/atomic_io), download (from datasets/_download), models (FrozenModel, ReportModel), markdown (table + number
  formatting with explicit missing marker), video, resample, log
2 stats (new): bootstrap (diagnosis/factorial), scene_bootstrap/SceneBlock/clopper_pearson/grouped_folds
  (calibration/stats), kendall (verification/sweep), Ci/Rate intervals, identical agreement/evaluation/timing stats
3 config   4 runtime: docker/udocker, heavy lock, runs (single new_run_dir + atomic run.json), environment
  (package_version from doctor), budget (from timing/budget)
5 physics (+measures/)
6 synth | sizing | generation(new: Wan2.1 generator + settings from timing/generation, timing/settings)
  sizing gets priors.py from extraction/providers/scale_size_prior + lookup_label
7 datasets (+scene_scope from calibration/scenes, select_clips from cli/extract)
8 checks | extraction (+load_error_profile from cli/extract)
9 verification: check_set (build_checks, checks_hash, active_checks_hash), scope, exclusion_record, runner, sweep,
  policies
10 diagnosis (+metrics/, verification_adapter = checks_port with injected deps)
11 calibration (+session from cli/calibrate)
12 human_agreement (was agreement) | selective_risk (new: from cli/abstain + verification/selective_risk)
13 finetune (+scorers/)
14 generator_comparison (was evaluation, +judges/) | timing (+targets/) | demo
15 reporting
16 app (composition root): plugins.py (PLUGIN_KINDS, ONLY caller of discover), context.py (AppContext), services.py
17 cli: main.py, context.py (typer glue), commands/<command>.py (argument parsing only)
— kubric_worker (py3.9; stdlib/numpy/schema only)
### A.2 Contract in pyproject [tool.importlinter] (layers list above; forbidden: kubric_worker & schema may not import
other physverify packages, pydantic, typer, torch). tests/test_architecture.py (stdlib ast) parses it, fails on any
upward or sibling import incl. lazy imports in functions. scripts/ may import physverify.app and below, never cli.
### A.3 Plugins: app/plugins.py PLUGIN_KINDS (MappingProxyType): checks, perturbations, families, providers, datasets,
container_tasks, metrics, policies, timing_targets, commands + NEW judges, scorers, truth_measures
(synth/truth_measures/<check>.py replacing MEASURE_KEYS/measure_all; validation test: every key is a registered check
id; gt_meta serialized sort_keys so bytes identical). Domain code receives registries/settings as arguments;
calibration/settings.check_registry() deleted. h3.py judge via registry("judges")[NAME]. Command plugin interface =
NAME + app (OWNER/IMPLEMENTED removed). Add-one-file table: check -> checks/rN_<what>.py; perturbation ->
synth/perturbations/<NAME>.py; perception model -> extraction/providers/<stage>_<name>.py; dataset ->
datasets/<name>.py (+optional configs/datasets/<name>.toml); metric -> diagnosis/metrics/<stage>_<metric>.py; judge ->
generator_comparison/judges/<name>.py; scorer -> finetune/scorers/<name>.py; command -> cli/commands/<name>.py.
Documented exception: a check with ground-truth grading adds synth/truth_measures/<check>.py.
### A.4 Reviewer proposals: accepted as in the table of the planner: move helpers down; generation package; commands
<=150 lines; split >400-line files (exemptions listed in ADR with reason); shared prereg mechanics only
(core/preregistration.py); selective_risk package (agreement's undecided policy stays); shared markdown primitives,
renderers stay with their study; runtime/budget; runtime.runs.new_run_dir; write_mp4 dedupe only if byte-identical;
_table/formatters merged keeping each missing marker; file_sha256/text_sha256; FrozenModel where identical config;
_checks_hash/parse_split/records_of/_torch/perturbation _check dedupe; Interval/Coverage dedupe only if field names and
dumps identical; NO_DECIDED NOT merged; dead code deleted (not_implemented, OWNER/IMPLEMENTED, exit 3,
collect.run_level, _windows centre_in_image/all_ids, runtime/threads.py or make docker.py use it); 11 test-only
functions -> tests/fixtures; agreement->human_agreement, evaluation->generator_comparison; reporting stays;
no calibration/protocol subpackage; check files keep id prefix (registry order); physics/measures renamed;
families/perturbations files keep serialized-key names; providers renamed <stage>_<name> (order guard);
tests mirror src; Sandbox x11 -> fixture; remove § and banners.
### A.5 One-off code: see M2. run_p1_perception.sh + test deleted. synth/hdri.py kept as synth/hdri_pools.py (derives
the hashed [dataset.hdri_pools] table). v1 ContinuityCheck -> tests/fixtures/reference/r4_jump_v1.py with its
helpers (goldens byte-identical). OWNER/IMPLEMENTED/not_implemented/NotImplementedCommandError/exit 3 deleted (grep
scripts for `exit 3`/`== 3` first).

## B. Naming
Rules: lower_snake nouns; no study/phase/iteration/ADR labels; `_` = private (not discovered, package-internal);
single-plugin module named after its serialized key (providers <stage>_<name>.py, gt providers gt_<stage>.py, checks
r<id>_<what>.py lower-cased id prefix preserving order); tests mirror src tests/<package>/test_<module>.py, script
tests in tests/scripts/, names state behavior; one config file per table family; scripts run_<stage>.sh,
<topic>_query.py, _<lib>.sh; CLI kebab-case; serialized names never change.
Packages: agreement -> human_agreement; evaluation -> generator_comparison; new stats, generation, selective_risk, app.
Modules (src): diagnosis/atomic_io -> core/atomic; core/hashing -> core/cache_keys; file hashers -> core/digests.
file_sha256; calibration/loader.sha256_of -> core/digests.text_sha256; datasets/_download -> core/download;
calibration/stats + diagnosis/factorial.bootstrap -> stats/bootstrap; verification/sweep.kendall -> stats/rank;
calibration/schema -> calibration/artifact (Ci/Rate -> stats/intervals); calibration/loader -> artifact_io;
devscore -> dev_score; heldout_h1 -> heldout_monotonicity; calibration/scenes -> datasets/scene_scope;
calibration/exclusion_record -> verification/exclusion_record; exclusions -> exclusion_rules; sealed + cli/_sealed ->
calibration/sealed_ledger + app/sealing; sigma -> sigma_recalibration; weights -> check_weights; settings.checks_hash/
check_registry -> verification/check_set; verification/selective_risk + cli/abstain logic -> selective_risk/{curve,
fit,report}; checks: r2_equilibrium -> r2_static_equilibrium, r3_collision_energy -> r3a_bounce_restitution +
r3b_collision_energy_gain, r4_integrity -> r4_missing_permanence + r4_size_constancy, r4_jump_noise -> r4_track_jump,
_common -> _results (VERIFY registry order unchanged: sort by file name!); physics/measures r1_penetration,
r2_support, r3_collision, r4_integrity, r4_jump_noise, r4_size_relative, r5_gravity -> penetration, support_overhang,
collision_energy, object_integrity, track_jump, relative_size, gravity_fit; synth: _measure -> truth_measures/ +
_magnitudes, _edit -> _state_edit, _windows -> _frame_windows, layout -> placement, scene_kit -> family_toolkit,
contacts_gt -> gt_contacts, kinematics_gt -> gt_velocity, hdri -> hdri_pools; providers depth_vda,
depth_vda_metric, detect_grounding_dino, track_sam2 -> depth_vda_small, depth_vda_metric_small,
detect_grounding_dino_tiny, track_sam2_1_hiera_tiny (programmer confirms keys); _attribution, _truth, _temporal,
_grounding, _lifting -> _impact_attribution, _conditional_truth, _temporal_smoothing, _grounding_per_query,
_backprojection (confirm content); diagnosis plans, chain, checks_port, cases, real_metrics, metrics/_states,
metrics/kinematics_derivs -> chain_variants, chain_runner, verification_adapter, measurement_cases,
real_video_metrics, _state_pairing, kinematics_derivatives; agreement pipeline, analysis, abstention, run_io, items,
decision -> comparison, policy_metrics, undecided_policy, verify_run_reader, comparison_items, clip_decision;
evaluation analysis, campaign, state, gate, bestofn -> preregistered_numbers, candidate_queries, prereg_state,
judge_gate, best_of_n; finetune scope, generate, scoring, signals, store, scorers/_status -> scorer_scope, candidates,
candidate_scoring, reward_hacking_signals, layout, _calibration_status; finetune/budget + timing/budget ->
runtime/budget; timing generation -> generation/wan21 (+settings); timing envinfo -> runtime/environment +
timing/environment; timing services -> target_services; timing mp4 -> core/video if byte-identical else mp4_writer;
sizing e2e, wire, draws -> end_to_end, protobuf_wire, objectron_draws; cli/commands/extract helpers -> datasets,
extraction, app/services; dataset.image_present -> runtime/container; doctor.package_version -> runtime/environment.
CLI (no aliases): h3 -> compare-generators; abstain -> selective-risk; data -> download; dataset -> synth; calibrate
devscore -> dev-score; results --a8v2-dir/--h1-evaluation/--h2-run/--abstention-run -> --chain-dir/
--calibration-evaluation/--agreement-run/--selective-risk-run. Blocked if the name is persisted (grep FREEZE_HINT,
RESCORE_HINT…).
Jargon identifiers (not serialized): _a7_factory -> verifier_factory; h3_step -> comparison_step; h1_report ->
calibration_report; h2_verdict() -> agreement_verdict(); h3_verdict() -> comparison_verdict(); heldout_h1() ->
heldout_monotonicity(); H1PrimaryView -> PrimaryScenesView; HeldoutH1Check -> HeldoutMonotonicity; H3Verdict ->
ComparisonVerdict (if not schema Record); neutral campaign_split messages.
Keep (S): h1_pass, meets_h1, heldout_h1 field, h1_primary, h2_verdict key, h2_group, h3_families_without_target,
h3_report.*, j3, j13, v1, "abstain curve", all *_SCHEMA/*_VERSION values, schema Record class names, check ids,
perturbation/family NAMEs, dataset names/subsets.
Scripts: run_h2.sh, h2_query.py -> run_human_agreement.sh, agreement_query.py; run_h3.sh ->
run_generator_comparison.sh; run_iteration.sh, iteration_arms.tsv -> run_perception_iteration.sh, perception_arms.tsv;
a8v2_query.py -> run_query.py; setup.sh -> setup_cuda.sh; deleted run_a8v2.sh, migrate_velocity.sh,
run_p1_perception.sh. pv()/query()/fq() defined once in _common.sh; freeze_query.py/run_query.py import app/domain,
never cli.
Script env vars: PHYSVERIFY_H2_* -> PHYSVERIFY_AGREEMENT_*; PHYSVERIFY_H2_A8V2_DIR -> PHYSVERIFY_AGREEMENT_CHAIN_DIR;
PHYSVERIFY_ITER_* -> PHYSVERIFY_ITERATION_*; PHYSVERIFY_FREEZE_P2_COMMIT -> PHYSVERIFY_FREEZE_EARLY_EXTRACTION_COMMIT;
PHYSVERIFY_A8V2_*, PHYSVERIFY_MIGRATE_* (unless kept by M2), PHYSVERIFY_P1_* removed. Unchanged: other
PHYSVERIFY_FREEZE_*, PHYSVERIFY_FINAL_* (incl V1_*), PHYSVERIFY_KUBRIC_*, PHYSVERIFY_H200_ROOT/ENV, UNSEAL*,
LOCK_HELD, CONFIG_DIR, UDOCKER*, ROOT, TIMING_THREADS, DEMO_OUT, FINETUNE_PILOT, AUTOEVAL_UPSTREAM, all PHYSVERIFY__*.
Tests: test_it1/it2_oracle_unchanged -> verification/test_oracle_golden.py; test_it2_temporal_survival ->
test_temporal_filter_keeps_detections; test_it3_contact_survival -> test_contact_filter_keeps_detections;
test_it4_* by behavior; test_it3_oracle_unchanged, test_it4_floor_refactor -> extraction/test_oracle_contacts_golden,
test_floor_contacts; checks/test_it4_v1_unchanged + r4_v1_golden_case -> checks/test_r4_jump_v1_reference;
calibration test_h1_*, test_heldout_h1, d9_golden_case -> test_monotonicity_*, test_heldout_monotonicity,
dev_score_golden_case (programmer: confirm what d9 golden actually pins and name it after that); golden
*_before_itN.json, d9_all_before_it1.json -> oracle.json, truth_contacts.json, <meaningful>.json via git mv (contents
untouched); tests/synth/data/f* -> data/clean/<family>; data/r5b -> data/gravity_scale; *_script.py/*_query.py tests
-> tests/scripts/; tests/unit split per package; class Sandbox x11 -> tests/fixtures/shell_sandbox.py; no_network/
_refuse x4 -> tests/conftest.py; test functions named after plan items (174 in 41 files) renamed after behavior.

## C. Config layout
configs/ default.toml, local.example.toml, [local.toml]
  verifier/ checks.toml check_scope.toml exclusion_rules.toml extraction.toml calibration.toml sealed.toml sizing.toml
            size_priors.toml size_model.toml(fitted)
  datasets/ kubric_synthetic.toml([dataset], was dataset.toml) kubric_fresh.toml physics_iq.toml physics_iq_views.toml
            objectron_static.toml pisa_real.toml (out of diagnosis.toml) videophy.toml (out of agreement.toml)
  studies/  diagnosis.toml primary_scenes.toml([h1_scope]) human_agreement.toml([agreement])
            generator_comparison.toml([h3*]) selective_risk.toml([abstention]) finetune.toml timing.toml demo.toml
Loader: config_dir.glob("*.toml") -> sorted(config_dir.rglob("*.toml")), skip local*.toml only at root; duplicate-table
error and doctor "source" show path relative to config_dir; merge semantics/env overrides/LOCAL_FILE unchanged.
Consumers: size.py MODEL_FILE -> verifier/size_model.toml; tests/conftest.py & tests/synth/conftest.py copytree
excluding local*; hygiene test rglob; tests opening configs/<name>.toml; run_final.sh (its v1-arm checks.toml copy:
C_V1 checkout has the OLD layout -> keep that path handling working for an old-layout checkout and note it); README.
Acceptance: raw-sections hash + key-path list unchanged; doctor effective values unchanged.

## D. Work packages (each ends: full pytest on server (count stated), ruff check, ruff format --check, mypy, E harness
empty diff, golden blob check)
WP0 – branch, ADR 0017 + Knowledge README index line, tests/contracts/ (a) registry key ORDER per kind literal lists,
  (b) set of schema.io.Record subclass names, (c) values of *_SCHEMA, *_VERSION, GT_REVISION, RUN_FILE, ledger and
  run.json field names, persisted file-name constants, (d) stable_hash(settings.sections) of repo configs + sorted
  flattened key paths, (e) model_json_schema()["properties"] keys of persisted pydantic models (calibration artifact/
  evaluation report, exclusion record, LedgerEntry, DatasetPlan, DatasetSettings, every check/provider Params,
  GenerationSettings, freeze manifest), (f) perturbation/family/dataset/policy/metric/provider keys. Later WPs may
  change only import lines of these tests (exception: WP5 provider-order guard). E harness built and baseline
  captured. Acceptance: 3446+N0 passed; `git diff 9eb5888 --stat -- src configs scripts` empty.
WP1 – dead code removal (A.5 + M2). Record collected counts of deleted test files in commit message. Acceptance:
  grep -rnE "migration_audit|audit-migration|migrate_velocity|run_a8v2|not_implemented|NotImplementedCommandError|
  \bIMPLEMENTED\b|\bOWNER\b|runtime\.threads|run_p1_perception|ContinuityCheck" src scripts README.md CLAUDE.md -> 0
  (ContinuityCheck only under tests/); GT_REVISION == 2 asserted.
WP2 – shared foundations/dedupe (core/{atomic,digests,download,models,markdown}, stats/, runtime/runs.new_run_dir 7
  sites, FrozenModel, constants, A.4 duplicates). Move verbatim first, then dedupe; pin old output bytes with a test
  before merging any formatter/table/serializer; synth/store keeps compact sorted JSON via schema.io.dumps (share only
  temp-file mechanism); save_atomic keeps .part resume semantics. Acceptance (count 1 each): def write_atomic|def
  atomic_write_bytes; def (sha256_of|file_sha256|_sha256) in src scripts (only core/digests); ConfigDict(frozen=True,
  extra="forbid") (only core/models); RUN_FILE = ; def (scene_bootstrap|bootstrap) only in stats/.
WP3 – composition root (app/{plugins,context,services}), PLUGIN_KINDS + AppContext moved, judges/scorers/
  truth_measures kinds, delete check_registry(), inject registries, remove upward imports (move select_clips,
  load_error_profile, image_present, package_version down), verification_adapter explicit deps, one patchable seam
  app.plugins.PLUGIN_KINDS (update monkeypatching tests). Acceptance: no `physverify.cli` import outside cli in src or
  scripts; `discover(` only in core/registry.py and app/plugins.py; no `Videophy2AutoevalJudge(` in cli; contract (a)
  unchanged.
WP4 – thin CLI: logic out of calibrate (Session -> calibration/session.py), finetune, h3 (H3Context ->
  generator_comparison/context.py), abstain (-> selective_risk), exclusions, timing, demo, diagnose, size, dataset,
  _stop_rule/_finetune_deps. Acceptance: no cli/*.py or cli/commands/*.py >150 lines; no cli/commands/_*.py; harness
  stdout/stderr byte-equal.
WP5 – package moves/renames, god-module splits, layer fix. Pure git mv commits with import fixes first, then split
  commits. Order guard: contract (a) literal-equal; provider renames may reorder `providers` kind only after proving
  registry("providers") is used only by key lookup and doctor output equal; then that assertion becomes sorted-set in
  the same commit with evidence. checks/metrics/policies order literal. Acceptance: tests/test_architecture.py passes;
  no src .py >400 lines except ADR exemptions; contract tests unchanged except imports.
WP6 – tests mirror src (git mv), fixture dedupe. Acceptance: every tests/<pkg>/ has src/physverify/<pkg>/ (except
  fixtures, scripts, contracts, docker); tests/unit absent; pass count equal to WP5; golden blob multiset equal.
WP7 – config layout (C). Acceptance: contract (d) unchanged; ls configs/*.toml -> default.toml, local.example.toml;
  harness doctor equal after mapping file names.
WP8 – CLI/scripts/env renames in one commit across scripts/h200/*, tests, README, env.example.sh,
  scripts/h200/README.md; single pv/query/fq. Acceptance: grep -rnE "PHYSVERIFY_(H2|ITER|A8V2|P1)_|physverify (h3|
  abstain|data|dataset) |pv (h3|abstain|data|dataset) |--a8v2-dir|--h1-evaluation|--h2-run" scripts src tests
  README.md -> 0; grep -c "^pv()" scripts/h200/*.sh total 1; tests/scripts green.
WP9 – language/comments/docstrings/jargon identifiers. 9.0 serial cross-file identifier renames (one commit). Then
  parallel batches over DISJOINT sets: (a) schema, core, stats, config, runtime, physics, kubric_worker, synth;
  (b) datasets, sizing, extraction, checks, verification; (c) diagnosis, calibration, selective_risk,
  human_agreement; (d) finetune, generator_comparison, timing, demo, reporting, app, cli; (e) tests; (f) configs,
  scripts, pyproject comments. Rules: comments only for physics formulas, axis conventions, reasons for non-obvious
  decisions; no banners, no §, no labels; docstrings say what, not which plan item; trim docstrings >6 lines to
  summary + non-obvious constraints (keep formulas, units, axis conventions, on-disk layouts); no history wording;
  stdout/stderr-only runtime strings neutral; persisted strings untouched. Integration: tests/test_process_labels.py
  (tokenize/ast): JARGON regex on comments+docstrings with NO allow-list; JARGON on string literals with literal
  allow-list; IDENT regex on identifiers with identifier allow-list; JARGON on configs/scripts text minus allow-list.
  JARGON = \bADRs?\b|\bADR[ -]?[0-9]{1,4}\b|\bA([3-9]|1[0-3])\b|\bA8 ?v2\b|\ba8v2\b|\bH[123]\b|\bJuicio\b|
  \bJ[0-9]{1,2}\b|\bplan-it[0-9]\b|\bit[1-5]\b|\bD[0-9]{1,2}(\.[0-9]+)?\b|\bAC-[0-9]+\b|\bT[0-9]{1,2}\b|
  [Pp]lan[- ]congelado|[Ee]nmienda|\bC_F\b|\bC_V1\b|\bWP-[A-Z0-9]+\b|\bP[1-5]\b|§|\b20[0-9]{2}-[01][0-9]-[0-3][0-9]\b
  IDENT = (^|_)(h[123]|a8v2|a[0-9]{1,2}|it[1-5]|d9|p[12])(_|$)
  BANNER = ^\s*#\s*(-{3,}|={3,})|^\s*#.*(-{5,}|={5,})\s*$
  PROCESS ATTRS = \b(OWNER|IMPLEMENTED|not_implemented|NotImplementedCommandError)\b
  Identifier allow-list (S): h1_scope, h1_pass, meets_h1, heldout_h1, h1_primary, h2_verdict, h2_group, h2_dev,
  h2_test, h3, h3_report, h3_families_without_target, j3, j13, PHYSVERIFY__H3__*.
  Literal allow-list: exact report fragments captured from renderers (file-scoped), "abstain curve", check ids,
  perturbation/family NAMEs, v1/v2 subsets. Excluded: tests/**/golden/**, tests/synth/data/**, *.json, *.jsonl,
  *.npz, uv.lock. Acceptance: test_process_labels passes; repo hygiene test passes; harness diff empty.
WP10 – enforcement: delete relaxed mypy overrides (measure error count first; split by layer if >~300); ruff adds
  RUF, ERA, PIE, RET, ARG, TID (banned-api physverify.cli/physverify.app outside cli/app/tests; ban relative imports),
  PTH, T20 (per-file-ignores for cli echo and scripts) — only rules that do not force behavior changes; architecture
  test reads [tool.importlinter]. Acceptance: no disallow_untyped_defs = false; mypy, ruff clean.
WP11 – docs: Repo/CLAUDE.md rewritten (rules, tree, layer rule, add-a-plugin table, contracts list; drop frozen-files
  and owners tables); README (no Activity column, new CLI names, env rename list, exit codes without 3, three studies
  named once in plain words); scripts/h200/README.md; ADR 0017 final exemptions. Acceptance: JARGON grep on README/
  CLAUDE.md finds only the single plain-words mention.

## E. Behavior-equivalence harness
Location: scratchpad/refactor/equivalence/ (run_equivalence.sh + driver.py). `git worktree add --detach` base 9eb5888
and head; same absolute scratch paths per side (config_hash includes paths); recreate run/{home,data,configs}; copy
side's configs without local*, plus local.toml with heavy lock under run/; `env -i HOME=… PATH=$PATH
PYTHONPATH=<wt>/src:<wt>/tests/fixtures <venv python> driver.py <scenario>`; move run -> out/<side>. Driver asserts
physverify.__file__ under <wt>/src; registers SyntheticFixtureDataset (tests/fixtures/synthetic_clip.py) by replacing
PLUGIN_KINDS["datasets"] in physverify.app.plugins if present else physverify.cli.context; calls
physverify.cli.main.run(argv, environ); records exit code, stdout, stderr. CLI names through old->new mapping.
Scenarios (CPU): S1 doctor; S2 verify --dataset synthetic_fixture --level oracle; S3 + --sweep; S4 dataset|synth
generate --subset smoke then list; S5 results on a chain fixture from tests/reporting/reporting_cases.py; S6+ every
other CPU CLI invocation that tests/*/test_cli*.py runs with committed fixtures (at least calibrate show/evaluate,
agreement compare if possible). Normalize only: run ids [0-9]{8}-[0-9]{6}-[0-9a-f]+ -> RUNID (dir names too); ISO
timestamps (created_utc, utc) -> TS; commit values -> COMMIT; argv mapped; doctor config-source paths mapped. Empty
diff required. Golden check: `git ls-tree -r <rev> -- tests | awk '$4 ~ /golden\/|synth\/data\//{print $3}' | sort`
identical at 9eb5888 and WP head.

## F. Commits
One+ commits per WP, English imperative `refactor(<area>): <what>`; body lists deleted tests with counts and order
evidence; git mv commits contain only import fixes; every commit builds; no tag ops, no push, no uv sync. Knowledge
lives outside the repo git.

## JUDGE AMENDMENTS (accepted by main thread; binding, override everything above)
J1. Sealed-ledger command labels are persisted (cli/commands/_sealed.py:45-60 GUARDED -> LedgerEntry.command,
    calibration/sealed.py:146): CLI may be renamed (selective-risk, dev-score) but each guarded command keeps a
    LEDGER_LABEL constant byte-identical to today ("abstain fit", "abstain curve", "calibrate devscore", ...). GUARDED
    strings go into contract (c).
J2. Parsed stdout contract: inventory every stdout/stderr line parsed by scripts/h200 (sed/grep/awk prefixes: "H2: ",
    "run: ", "evaluation: ", "preregistration: ", "record: … sha256", "  gate: passed", "real dataset … skipped",
    "N scenes, N clips,", "docker_image:", "container_engine:", "source: --set", the
    "^(note: |oracle/|perceived/|  R|selection )" filter, doctor lines used by run_freeze.sh, ...). Pin them in
    tests/contracts/test_parsed_output.py; exempt from WP4/WP9 rewording (literal allow-list). Add a test: every
    literal sed/grep prefix in scripts/h200 still occurs in src.
J3. scripts/h200/freeze_query.py DIFF_PATHS (j3, j13) must be updated to the new layout (j13: extraction package +
    configs/verifier/extraction.toml; j3: every package/config that holds verifier code after the moves: checks,
    physics, extraction, verification, core/cache_keys, stats, sizing/priors, datasets/scene_scope, ...). ADR 0017
    records that iteration/early-extraction artifacts from pre-refactor commits cannot be adopted by post-refactor
    freezes (run_freeze.sh:149) and later rounds must run on post-refactor commits.
J4. run_final.sh v1 arm (449-468, 497-501) runs the C_V1 checkout (old flat loader) with HEAD-built configs:
    v1_configs must flatten the nested layout into one flat dir (file names unique; then overlay C_V1 checks.toml).
    Test it in tests/scripts.
J5. Config consumers also: scripts/h200/setup_autoeval.sh:3,15-19 (sed on configs/h3.toml); doc strings naming
    config files (verification/runner.py:36 PROVISIONAL_SOURCE — check whether persisted before changing).
J6. Extra layer violations to fix: extraction/plan.py:110-111 (inject check registry; move CHECKS_SECTION <= L8);
    diagnosis/scene_stats.py:19 (calibration.table.scene_of -> move scene_of down); finetune/generate.py:36
    (timing.targets.wan21_generate.batch_frames -> generation/); demo/settings.py:11 (timing.settings.SLUG_PATTERN
    -> lower shared place). The "commands" registry stays in cli (cli discovers commands; app/plugins holds every
    other kind).
J7. Contract-test permitted changes, explicit: commands keys/order may change in WP8; providers order in WP5 (with
    evidence); new kinds judges/scorers/truth_measures added in WP3. All else literal.
J8. Greps fixed: WP1 `\bContinuityCheck\b`, `physverify\.runtime\.threads|runtime/threads\.py`; WP2 also matches
    ConfigDict(extra="forbid", frozen=True) in either order.
J9. Exception class names persisted (diagnosis/report.py:146, cli/commands/extract.py:82 type(error).__name__):
    all PhysverifyError subclass names pinned in contract (b); none renamed.
J10. Harness: (a) each scenario asserts expected exit code and non-empty output file set; (b) WP0 pins literal hex
    of config_hash, look_config_hash, checks_hash, DatasetSettings.content_hash and stage_key of every provider on a
    fixture clip (contract test); (c) tool scratchpad/refactor/equivalence/ast_equal.py: for pure-move (WP4a) and
    WP9 comment/docstring commits, ast.dump with docstrings stripped is equal per module (mapped by rename), except
    a listed TSV of allowed changed literals/identifiers; (d) truth_measures equivalence test: old measure_all vs
    plugins on tests/synth/data fixtures, byte-equal; (e) `--help` of every command captured (help-text diffs
    reviewed, allowed only where intended).
J11. Sequencing: WP4a = pure package/module renames (git mv + import fixes, AST-equal) BEFORE logic moves; WP4b = thin
    CLI; WP5 = splits of oversized modules + remaining layer fixes + tests/test_architecture.py. WP9 batches run in
    separate git worktrees from the 9.0 commit; each checked with git diff --name-only against its file set.
J12. IDENT regex: (^|_)(h[123]|a8v2|a([3-9]|1[0-3])|it[1-5]|d9)(_|$); no bare a0..a2/p1/p2.
J13. Check file rule: r<lower id>_<what>, sorted file order == check order; R4_jump is r4_track_jump. Contract (a)
    enforces.
J14. freeze_query.py library logic moves into the package (e.g. physverify/freeze/ or app layer, decide) and
    scripts/h200/*.py come under mypy (tool.mypy files) ; scripts only parse args and call.
J15. WP11 also updates Knowledge/project/contexto.md (Estado) and Knowledge/README.md index — main thread does this.
Suggestions adopted: judge selected by a config-model default field (not hard-coded); time_dependent_checks default
derived from a check attribute (config key kept, value identical); add-one-file table lists every kind; keep
_steps.sh coverage when test_run_a8v2_script.py is deleted; WP10 ARG/PTH not forced on plugin protocol signatures.
