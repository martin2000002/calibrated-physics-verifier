# Language/comments review + naming review (read-only reviewers, 2026-10-22)

## Language and comments (scanner: scratchpad/scan.py)
- No accented chars anywhere. Non-ASCII: `§` (90 uses / 78 files, "ADR 00xx §n").
- Spanish process words: `Juicio` 34 lines/30 files (agreement/decision.py:1, agreement/prereg.py:1,
  configs/finetune.toml:178, tests/calibration/test_sealed.py:139); `plan-congelado` 27/24 (scripts/h200/run_freeze.sh:2,
  freeze_query.py:2); `Enmienda` 6/4 (configs/exclusion_rules.toml:1, scripts/h200/run_h2.sh:91 inside a die message);
  `Entregable` 3 (configs/checks.toml:3, timing.toml:3, diagnosis.toml:1).
- Knowledge paths: configs/extraction.toml:36,39; scripts/h200/run_final.sh:72; CLAUDE.md:3-4 points to Spanish ADR.
- Thesis-process references (dominant problem): ADR 00xx 335 lines/263 files; activity ids A4..A13 204/93; H1/H2/H3
  192/92; Juicio/Jn 176/79; it1..it5/plan-itN 150/64; Dn 133/91; AC-n 99/48; plan-congelado/Fn 60/27; Tn 46/28;
  C_F/C_V1 15/4; dates 12/11. 330/1933 src docstrings, 249/699 test docstrings; 88 runtime strings in 46 src files
  (datasets/objectron_static.py:255 "(J14)", diagnosis/real_video.py:42 "A7's check suite",
  calibration/settings.py:43 "before 2026-10-20").
- Jargon in identifiers: OWNER="A9" etc. required attribute of every CLI subcommand (20 files; shown in stub exit
  messages); CLI subcommand `h3`; config [h1_scope], [h3], [h3.gate], h2_dev, h2_test; src h1_pass (16), h2_verdict (14),
  meets_h1, heldout_h1, h3_report, a8v2_dir, _a7_factory; 43 test functions like test_t8_t14_the_v1_results_do_not_move.
- Banners `# --- x ---`: 114 (src 38, tests 72, scripts 4). Worst: tests/extraction/test_impacts.py (8),
  tests/checks/test_r4_jump_noise.py (7), tests/verification/test_it3_contact_survival.py (6), cli/commands/calibrate.py
  (5), calibration/exclusions.py (5).
- History wording ~44 ("no longer", "legacy", "before itN", "(v1)…(v2)" in checks/r4_integrity.py:107,201).
- No commented-out code, no TODO/FIXME.
- Docstrings: 2658 total; 270 are >=4 lines, 50 >=8. Longest: finetune/store.py:1 (25, layout useful),
  freeze_query.py:1 (22), extraction/providers/_impacts.py:1 (20), physics/equilibrium.py:1 (15).
- Module docstrings: src all except 30 __init__.py; tests 66/278 missing (all tests/unit).
- Top files: scripts/h200/README.md, README.md, tests/extraction/test_impacts.py, tests/checks/test_r4_jump_noise.py,
  run_final.sh, synth/migration_audit.py, CLAUDE.md, run_freeze.sh, run_h2.sh, tests/timing/test_run_final_script.py.

Rules: (1) delete process citations ((ADR nnnn…), §n, Juicio, Jn, plan-itN, itN, Tn, AC-n, Fn, Pn, Enmienda,
Entregable, dates); rewrite as the rule it stands for if needed. (2) where the ID is the only content, spell the
meaning. (3) delete banners. (4) delete history wording; describe current behaviour; goldens say "reference outputs
recorded at commit X". (5) trim docstrings >6 lines to summary + non-obvious constraints; keep formulas, units, axis
conventions, on-disk layouts. (6) rename jargon identifiers. (7) neutral runtime strings. (8) README/CLAUDE.md:
define study names once in words. (9) one-line module docstrings for test files lacking one.

## Naming review
Verdict: plugin layout sound; "dump" feeling from names. Causes: process ids in names (R-codes, f-codes, it1-it4,
d9, h1/h2/h3, a8v2, p1); hypothesis-named packages (agreement=H2, evaluation=H3, CLI h3, configs/h3.toml); generic
module names colliding across packages (chain, layout, weights, scope, attribution, budget, evaluate, schema/loader);
tests/ not mirroring src/.
Lying names: checks/r4_integrity.py:200 dead v1 ContinuityCheck; configs/diagnosis.toml holds dataset sections;
diagnosis/atomic_io.py general utility; synth/layout.py (placement) vs evaluation/layout.py (paths);
configs/h1_scenes.toml holds [h1_scope]; provider files vs keys (depth_vda.py->vda_small,
detect_grounding_dino.py->grounding_dino_tiny, depth_vda_metric.py->vda_metric_small).

CONTRACTS (keep string or alias):
- Check ids R1..R5b, R4_missing/size/jump: goldens, run tables, calibration artifacts, thesis. KEEP ids; rename files.
- Perturbation/family NAMEs (r5b_gravity_scale, f2_bounce…) in spec.json/perturbation.json, manifests, clip ids. KEEP.
- CLI names (h3, abstain, data, dataset, scene-stats, size…) used by scripts/h200/*.sh via pv.
- Env vars PHYSVERIFY__H3__AUTOEVAL__ROOT (11), PHYSVERIFY__KUBRIC_FRESH__*, PHYSVERIFY__SEALED__*,
  PHYSVERIFY__OBJECTRON_STATIC__*, PHYSVERIFY__AGREEMENT__RULE.
- Dataset names kubric_fresh*, subset h2_dev (datasets/_videophy_tables.py:94) in sealed ledger and run dirs.

Rename proposals (subset; old -> new):
packages: agreement -> human_agreement; evaluation -> generator_comparison (CLI h3 -> compare-generators,
configs/h3.toml -> generator_comparison.toml); reporting -> results; calibration protocol modules (devscore, sealed,
exclusions, exclusion_record, scenes, heldout_h1) -> protocol/.
checks: r1_interpenetration -> interpenetration; r2_equilibrium -> static_equilibrium; r3_collision_energy -> split
bounce_restitution (R3a) + collision_energy_gain (R3b); r4_integrity -> object_permanence + size_constancy (v1
ContinuityCheck out); r4_jump_noise -> track_continuity; r5a_fall_shape -> fall_shape; r5b_gravity_value ->
gravity_value; _common -> _results.
physics/measures (FROZEN): r1_penetration -> penetration, r2_support -> support_overhang, r3_collision -> collision,
r4_integrity -> integrity, r4_jump_noise -> jump_over_noise, r4_size_relative -> relative_size, r5_gravity -> gravity;
tracks.py -> table_rows.py.
synth families f1..f5 -> projectile, bounce, air_collision, stack, rolling_contact; perturbations: r1_penetration ->
push_into_neighbour, r2_unsupported_rest -> unsupported_rest, r3_energy_gain -> collision_energy_gain,
r3_restitution_gain -> bounce_restitution_gain, r4_resize/teleport/vanish -> resize/teleport/vanish,
r5a_gravity_switch -> gravity_switch, r5a_object_gravity -> object_gravity, r5b_gravity_scale -> gravity_scale,
none -> identity; _measure -> _truth_magnitudes; _edit -> _state_edit; _windows -> _frame_windows; layout ->
placement; scene_kit -> family_toolkit; contacts_gt -> gt_contacts; kinematics_gt -> gt_velocity.
extraction/providers: depth_vda -> depth_vda_small; depth_vda_metric -> depth_vda_metric_small;
detect_grounding_dino -> detect_grounding_dino_tiny; _attribution -> _impact_attribution; _truth ->
_conditional_truth; _temporal -> _temporal_smoothing; _grounding -> _grounding_per_query; _lifting -> _backprojection.
diagnosis: atomic_io -> core; plans -> chain_variants; chain -> chain_runner; checks_port -> verification_adapter;
cases -> measurement_cases; real_metrics -> real_video_metrics; metrics/_states -> _state_pairing;
kinematics_derivs -> kinematics_derivatives.
calibration: schema -> artifact_schema; loader -> artifact_loader; weights -> check_weights; sigma ->
sigma_recalibration; devscore -> dev_score; heldout_h1 -> heldout_evaluation; scenes -> primary_scenes; sealed ->
sealed_ledger; exclusions -> exclusion_rules.
agreement: pipeline -> comparison; workflow -> freeze_and_compare; analysis -> policy_metrics; abstention ->
undecided_policy; run_io -> verify_run_reader; items -> comparison_items; decision -> clip_decision.
evaluation: analysis -> preregistered_numbers; campaign -> candidate_queries; state -> prereg_state; gate ->
judge_gate; bestofn -> best_of_n.
finetune: scope -> scorer_scope; generate -> candidates; scoring -> candidate_scoring; signals ->
reward_hacking_signals; store -> layout; scorers/_status -> _calibration_status; budget collides with timing/budget.
timing: envinfo -> environment; generation -> wan21_generator; mp4 -> mp4_writer; services -> target_services.
sizing: e2e -> end_to_end; wire -> protobuf_wire; draws -> objectron_draws.
cli/commands: h3 -> compare_generators (C); abstain -> selective_risk (C); data -> fetch_data (C); dataset ->
synthetic_dataset (C); _finetune_deps/_runs/_sealed/_stop_rule -> non-plugin shared package.
configs: h1_scenes.toml -> primary_scenes.toml; h3.toml -> generator_comparison.toml; abstention.toml ->
selective_risk.toml; dataset sections out of diagnosis.toml/agreement.toml into datasets file(s).
scripts/h200: run_a8v2.sh -> run_calibration_chain.sh (+a8v2_query.py); run_h2.sh -> run_human_agreement.sh
(+h2_query.py); run_h3.sh -> run_generator_comparison.sh; run_p1_perception.sh -> run_perception_configs.sh;
run_iteration.sh -> run_perception_iteration.sh; iteration_arms.tsv -> perception_arms.tsv; setup.sh -> setup_cuda.sh.
tests: mirror src; new tests/scripts/ for all *_script/*_query tests; tests/datasets/ for dataset tests scattered in
agreement/diagnosis/sizing/synth/extraction; split tests/unit into per-package dirs; rename test_itN_*, test_h1_*,
d9_golden_case, golden/*_before_itN.json, tests/synth/data/f*_ -> data/clean/<family>, data/r5b -> data/gravity_scale.
