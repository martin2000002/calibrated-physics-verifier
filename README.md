# physverify

Calibrated physics verifier for generated video of rigid bodies. Given a clip, `physverify`
extracts object states and contacts (from ground truth or from perception models), evaluates
physics checks R1-R5b (interpenetration, support equilibrium, collision energy, object integrity,
free-fall shape and gravity value) and reports each verdict with the propagated measurement error,
abstaining when the error is too large to decide.

It is part of an undergraduate thesis (Computer Science, USFQ) built on three studies: the
calibration of each check against violations of known size injected into synthetic scenes, the
agreement of the verifier with human physics labels on generated video (VideoPhy) compared with a
reference vision-language model, and the comparison of a video generator fine-tuned with the
verifier as preference signal against best-of-8 sampling from the base generator.

## Status

Every subcommand is implemented. The library code has unit tests, and most subcommands are also
invoked by a test of their own. The checks are calibrated on the calibration split of the synthetic
dataset only; `calibrate publish` writes the thresholds that `verify` uses, and the values in
`configs/verifier/checks.toml` are provisional fallbacks for when none is published. A published
threshold table carries the `checks_hash` of the check parameters and variants it was fitted with,
and `verify` refuses (exit 2) a table without that hash or with another one. A provisional
`verification.thresholds.<id>` is read in the unit of the check's default implementation; a variant
that reports another unit needs `verification.threshold_units.<id>` set to that unit next to its
tau. `configs/verifier/sealed.toml` lists the sealed sets: a command refuses to compute on one
unless it is unsealed explicitly, and each look is logged before any number is computed. The final
test (`scripts/h200/final_run.sh`) makes its looks in a clean checkout at the frozen commit, which
its `freeze.json` records with the `configs/` tree, the profile, the device and the sha256 of each
file a look judges with. The tag `final-freeze` marks an earlier freeze; the ledger lines written at
its commit name a bare dataset, and `configs/verifier/sealed_legacy.toml` maps them to the dataset
version each one read.

The calibration artifact is a run output and is not versioned. The size model is versioned
(`configs/verifier/size_model.toml`): its coefficients are estimates and carry their measured
error. Generation, fine-tuning and the generator comparison need a CUDA GPU; the demo generates a
video on the GPU or scores a given video on the CPU.

| Command | What it does |
|---|---|
| `timing` | Per-clip cost harness (generation, extraction, checks, rendering, DPO step) and GPU budget; only GPU runs with `--device cuda --profile precise` count as measurements |
| `synth` | Kubric synthetic dataset: five scene families, ten perturbation types at four magnitude levels, per-stage ground truth; resumable steps and ground-truth migrations |
| `extract` | Eight-stage perception pipeline (detection, tracking, depth, camera, 3D lifting, scale with provenance, kinematics, contacts) with a `fast` and a `precise` profile, which differ in providers and sampling (`configs/verifier/extraction.toml`), on the device given by `--device`; weight download and cache purge |
| `diagnose` | Extractor error per stage, end to end and by factorial attribution, plus real video (Physics-IQ, PISA) |
| `download` | Real datasets (Physics-IQ, PISA) and, by name, the sealed real controls |
| `verify` | Checks R1-R5b (nine measures) on ground-truth ("oracle") or perceived states, with verdict intervals, abstention and a specificity sweep |
| `calibrate` | Calibration on the calibration split: thresholds, false-positive rates, detection limits, per-check and per-stage sigma correction, clip score; held-out and real-video evaluation; publication of the thresholds |
| `exclusions` | Applies the pre-registered exclusion rules to the calibration results and writes the hashed decision record the later studies read |
| `selective-risk` | Risk-coverage of the threshold policy: abstention level chosen on the calibration split, held-out curve |
| `size` | Per-class object size model learned on Objectron: fetch, fit, held-out and end-to-end error |
| `agreement` | Verifier vs reference VLM on VideoPhy human labels: data, reviewed subset, pre-registration on dev, comparison on test |
| `finetune` | Prompt set, Wan2.1-T2V-1.3B candidates scored by the verifier, preference pairs with a pilot stop rule, Flow-DPO LoRA training and checkpoint selection |
| `compare-generators` | Fine-tuned vs base vs best-of-8 under the verifier and VideoPhy-2 AutoEval, with a pre-registration and a regression gate for the judge |
| `demo` | One prompt or video through the whole verifier: per-check report and overlay video |
| `scene-stats` | Per-clip detection and depth errors, per-scene tables and scene-bootstrap differences between two extractor configurations |
| `results` | Collects the reported numbers of a calibration chain (the stored output directory of a calibration-chain runner that is no longer in the repository) and an agreement run into `results.md` and `results.json`, each value with its source and estimates marked |
| `final-table` | The final run's checks and closing table: no weight-0 check can make a clip abstain (`preflight`), every active check of an evaluation got its input (`inputs`), and `final_table.{md,json}` with one row per closing criterion (`build`) |
| `report` | Publication figures of a finished final run: one grayscale vector PDF per figure from the outputs the run wrote, texts from a labels TOML (`figures`; needs matplotlib, see the `figures` extra) |
| `doctor` | Effective configuration, paths, lock and package versions |

Runs on the university H200 use the scripts in `scripts/h200/`: on the GPU, timing, fine-tuning,
the generator comparison and demo generation; on the CPU, Kubric rendering, the perception
iterations and the human-agreement study (both with CUDA optional). The freeze and the final looks
run on the device of the adopted iteration (`PV_DEVICE`, recorded in `freeze.json`), and
`scripts/h200/final_run.sh` runs the whole final test in one command. See
[its README](scripts/h200/README.md) and each script's header.

## Installation

Requires Linux, [uv](https://docs.astral.sh/uv/) (Python 3.12) and, for synthetic data, Docker or
udocker with the pinned Kubric image (`runtime.docker_image` in `configs/default.toml`; the engine
is set in the `[container]` table, see `configs/local.example.toml`).

```sh
uv sync                                   # default groups: dev, cpu (CPU PyTorch), perception, data
uv sync --no-default-groups --group dev   # light: core + dev tools only
uv sync --no-default-groups --group dev --group cu128 --group perception --group generation --group data   # CUDA 12.8
uv sync --no-default-groups --group dev --extra figures   # light, plus matplotlib for `report figures`
cp configs/local.example.toml configs/local.toml   # machine-local overrides (ignored by git)
```

`generation` (diffusers, peft) is only needed for Wan2.1 generation and fine-tuning. The `figures`
extra (matplotlib) is only needed for `report figures`; the `perception` group already brings
matplotlib, so the default install draws them too. `cpu` and `cu128` are conflicting groups with
their own PyTorch indexes; `uv.lock` covers both. Use `uv run --no-sync` so uv does not install the
default groups on its own; installs are heavy jobs and run under the lock
(`flock <lock> uv sync ...`).

## Usage

```sh
uv run --no-sync physverify --help
uv run --no-sync physverify <command> --help
uv run --no-sync physverify doctor
uv run --no-sync physverify --profile precise --device cuda doctor
uv run --no-sync physverify --set runtime.threads=2 doctor
```

Global options go before the subcommand: `--config-dir` (default `$PHYSVERIFY_CONFIG_DIR` or
`configs/`), `--profile fast|precise`, `--device cpu|cuda|cuda:N`, `--set section.key=value`
(repeatable), `--no-lock`, `--lock-timeout SECONDS`, `--log-level`. Configuration is layered,
lowest to highest: model defaults, `configs/**/*.toml` (except the root's `local*.toml`), the
active profile's `[<section>.profiles.<name>]` tables, `configs/local.toml`,
`PHYSVERIFY__SECTION__KEY` environment variables, `--device` / `--profile`, then `--set`. `doctor`
prints the main effective values (profile, device, threads, paths, container, lock), the
configuration hash and the layers it read (`source:` lines: each config file, then each
`PHYSVERIFY__` variable, `--device` and each `--set` given). A bad option value, a missing config
directory, an unknown section in `--set` or the environment, a `<section>.profiles` key above the
config files (use `--profile`), or an invalid field exits 2; each invalid field prints one
`[section] key: message (got value)` line.

`configs/` holds `default.toml` and `local.example.toml`; `verifier/` the verifier's own settings
(checks and their scope, extraction, calibration, exclusion rules, size priors and model, sizing,
sealed sets), `datasets/` the settings of each dataset that has some (`kubric_fresh.toml` serves
`kubric_fresh`, `kubric_fresh_ood` and `kubric_fresh_blind`), `synth/` the synthetic object size
distribution and `studies/` one file per study. Each top-level table lives in exactly one file.

Exit codes: 0 ok, 1 error, 2 usage, configuration or resource budget, 75 heavy lock busy.

### Renamed names

Older runs, notes and launchers may use these names; the old names are no longer read.

| Old | Current |
|---|---|
| `physverify h3` | `physverify compare-generators` |
| `physverify abstain` | `physverify selective-risk` |
| `physverify data` | `physverify download` |
| `physverify dataset` | `physverify synth` |
| `physverify calibrate devscore` | `physverify calibrate dev-score` |
| `results` options `--a8v2-dir`, `--h1-evaluation`, `--h2-run`, `--abstention-run` | `--chain-dir`, `--calibration-evaluation`, `--agreement-run`, `--selective-risk-run` |
| `PHYSVERIFY_H2_*` | `PHYSVERIFY_AGREEMENT_*` |
| `PHYSVERIFY_H2_A8V2_DIR` | `PHYSVERIFY_AGREEMENT_CHAIN_DIR` |
| `PHYSVERIFY_ITER_*` | `PHYSVERIFY_ITERATION_*` |
| `PHYSVERIFY_FREEZE_P2_COMMIT` | `PHYSVERIFY_FREEZE_EARLY_EXTRACTION_COMMIT` |
| `scripts/h200/setup.sh` | `scripts/h200/setup_cuda.sh` |
| `scripts/h200/run_h2.sh`, `h2_query.py` | `scripts/h200/run_human_agreement.sh`, `agreement_query.py` |
| `scripts/h200/run_h3.sh` | `scripts/h200/run_generator_comparison.sh` |
| `scripts/h200/run_iteration.sh`, `iteration_arms.tsv` | `scripts/h200/run_perception_iteration.sh`, `perception_arms.tsv` |
| `scripts/h200/a8v2_query.py` | `scripts/h200/run_query.py` |
| `scripts/h200/run_a8v2.sh` (`PHYSVERIFY_A8V2_*`) | Removed. Its stored output directory, a calibration chain, is still read by `results --chain-dir` and `run_human_agreement.sh`; new runs use `run_freeze.sh` and its `freeze.json` |
| `scripts/h200/run_p1_perception.sh` (`PHYSVERIFY_P1_*`) | Removed. Perception configurations run as declared arms of `scripts/h200/run_perception_iteration.sh` (`perception_arms.tsv`) |
| `scripts/h200/migrate_velocity.sh` (`PHYSVERIFY_MIGRATE_*`) | Removed. `physverify synth migrate`, then `synth build` and `extract purge` by hand (see the H200 README, Ground-truth migrations) |

Persisted names (record classes, schema ids, config tables and keys, ledger labels) did not
change, so existing runs, caches and artifacts stay readable.

## Tutorial

This walk-through goes from a fresh checkout to verified synthetic clips and a scored real video,
all on the CPU. The installation, `doctor` and the oracle run of step 2 were run on a laptop
(4 threads, 7.6 GB of RAM, no NVIDIA GPU), on clips rendered on the H200 and copied over. The
container steps, rendering, perception, the demo and calibration were run on the CPU of the
university H200 (GPUs hidden, the default 4 threads) with `python -m physverify` from the
server's own environment, whose PyTorch build differs from `uv.lock`. Times are indicative (the
server is shared) and output is shortened: `...` stands for an absolute path or omitted fields,
`<run id>` for the timestamped run name.

### 1. Install and check the configuration

The light install is enough for `doctor` and for the oracle run of step 2:

```sh
uv sync --no-default-groups --group dev
uv run --no-sync physverify doctor
```

It reports `torch: not installed` and, while no `configs/local.toml` sets a lock,
`heavy_lock: disabled (runtime.heavy_lock is empty)`. Perception needs the default groups (CPU
PyTorch, the perception models, the data tools). Installing them is a heavy job, so it runs under
the lock that `configs/local.toml` names:

```sh
cp configs/local.example.toml configs/local.toml
flock "$XDG_RUNTIME_DIR/physverify-heavy.lock" uv sync
uv run --no-sync physverify doctor
```

```text
python: 3.12.16
numpy: 2.5.3
torch: 2.14.0+cpu
transformers: 5.17.0
profile: fast
device: cpu
threads: 4
workers: 1
config_dir: .../configs
source: .../configs/datasets/kubric_fresh.toml
...
source: .../configs/local.toml
config_hash: 6f3e19388e3f
paths.data_dir: .../data
...
paths.error_profile: .../data/calibration/stage_errors.fast.json
paths.thresholds: .../data/calibration/thresholds.json
...
container_engine: docker
heavy_lock: /run/user/1000/physverify-heavy.lock (free)
```

The `source:` lines are the configuration layers in the order they were read, `configs/local.toml`
last. `config_hash` is the hash of the effective settings, which every run records in its
`run.json`. The `paths.*` lines say where data, caches, runs and calibration artifacts go (all
under `data/` by default). Profile and device default to `fast` and `cpu`. Synthetic data needs
the pinned Kubric image; the probe starts it and imports what the renderer uses:

```sh
uv run --no-sync physverify synth pull-image
uv run --no-sync physverify doctor --probe-container
```

```text
...
container_probe: ok (kubric, bpy, pybullet, physverify.kubric_worker.build_scene)
```

### 2. Verify synthetic clips

```sh
uv run --no-sync physverify synth build --subset local
uv run --no-sync physverify synth list --subset local
```

```text
generate     7 done    0 failed    0 pending
simulate     7 done    0 failed    0 pending
apply       23 done    0 failed    0 pending    0 skipped
perturb     23 done    0 failed    0 pending    0 skipped
render      23 done    0 failed    0 pending    0 skipped
export      23 done    0 failed    0 pending    0 skipped
render: 0.57 s per frame (measured)
```

The `local` subset is one scene per family (projectile, bounce, collision, stack, contact) and one
mid level of every perturbation that applies to it: 23 clips of 24 frames at 24 fps and 128x128
pixels, from 7 simulations (each gravity-scale clip is simulated again under its own gravity).
Each clip is written to `data/synthetic/v1/clips/<clip id>/` with its video, its ground-truth
states and contacts, and `record.json`, the injected perturbation with its magnitude in the unit
of the check it targets. The build resumes: a step that fails writes `failed.json` in its item,
the command exits 1, and running it again does only what is missing. Here two of the seven
simulations failed on the first build (397 s) and passed on the second (306 s); the
`Error: Not freed memory blocks` lines the simulator prints do not fail it.

Each clip's `manifest.json` gives its split. In `local` the 9 projectile and collision clips are
the calibration split and the other 14 the evaluation split, which `configs/verifier/sealed.toml`
seals (`kubric_synthetic:eval`): a command that would compute on a sealed clip refuses with
`... nothing was computed` unless the set is unsealed explicitly, which spends its single look
(step 4). The commands below therefore name four calibration clips with `--clips`; without it,
`verify` would select the sealed clips too and refuse. The four are a clean projectile scene, the
same scene under gravity 20% weaker (`r5b_gravity_scale_L3`), a clean two-ball collision, and the
same collision with 50% more kinetic energy after impact, momentum conserved
(`r3_energy_gain_L3`). With ground-truth states (`--level oracle`):

```sh
uv run --no-sync physverify verify --dataset kubric_synthetic --subset local --level oracle \
    --clips f1_projectile-s000__none --clips f1_projectile-s000__r5b_gravity_scale_L3 \
    --clips f3_collision-s000__none --clips f3_collision-s000__r3_energy_gain_L3
```

```text
run: .../data/runs/<run id>
thresholds: provisional
R1: scored 4 abstained 0 no_event 0 error 0 FAIL 0
R2: scored 0 abstained 0 no_event 4 error 0 FAIL 0
R3a: scored 4 abstained 0 no_event 0 error 0 FAIL 0
R3b: scored 2 abstained 0 no_event 2 error 0 FAIL 1
R4_missing: scored 4 abstained 0 no_event 0 error 0 FAIL 0
R4_size: scored 4 abstained 0 no_event 0 error 0 FAIL 0
R4_jump: scored 4 abstained 0 no_event 0 error 0 FAIL 0
R5a: scored 4 abstained 0 no_event 0 error 0 FAIL 0
R5b: scored 4 abstained 0 no_event 0 error 0 FAIL 1
```

The run directory holds `run.json` (command line, commit, profile, device, `config_hash`),
`checks_hash.txt` (a hash of the version and effective parameters of every check), `table.csv`
(one row per clip and check) and `checks.jsonl` (one result per subject):

```text
clip_id,check_id,status,magnitude,sigma,verdict,abstain_reason,n_subjects,n_abstained,unit,row_step
f1_projectile-s000__r5b_gravity_scale_L3,R3a,scored,-0.2639661185915285,0.0,ABSTAIN,TOO_FEW_FRAMES,2,1,e_minus_1,1
f1_projectile-s000__r5b_gravity_scale_L3,R5a,scored,3.9025825156954706e-14,0.0,PASS,,3,0,accel_rel_diff,1
f1_projectile-s000__r5b_gravity_scale_L3,R5b,scored,0.19999999494484558,0.0,FAIL,,1,0,g_rel_error,1
f3_collision-s000__none,R3b,scored,-0.07584530966600136,0.0,PASS,,1,0,dE_over_E,1
f3_collision-s000__r3_energy_gain_L3,R3b,scored,0.4999999822398659,0.0,FAIL,,1,0,dE_over_E,1
```

```text
{"data":{"abstain_reason":null,"check_id":"R3a","clip_id":"f1_projectile-s000__r5b_gravity_scale_L3","details":{"e":0.7360338814084715,...},"frame_span":[9,22],"input_level":"oracle","magnitude":-0.2639661185915285,...,"sigma":0.0,"subject":["ball_1","__env__"],"threshold":0.05,"unit":"e_minus_1","verdict":"PASS"},...}
{"data":{"abstain_reason":"TOO_FEW_FRAMES","check_id":"R3a",...,"details":{"row_step":1,"rows_after":2,"rows_before":6},"frame_span":[20,21],...,"magnitude":null,...,"sigma":NaN,"subject":["ball_0","__env__"],"threshold":null,...,"verdict":"ABSTAIN"},...}
```

A subject is what a check judges: an object, a pair, or an object with the environment
(`__env__`). Each gets a magnitude m in the check's unit, its sigma (the propagated measurement
error: 0 for ground truth, NaN while unknown) and a verdict against the threshold tau, with
k = 2 (`k_sigma` in `[verification.policies.threshold]`): FAIL when m - k sigma > tau, PASS when
m + k sigma <= tau, and ABSTAIN when the interval m +/- k sigma contains tau
(`INTERVAL_CROSSES_THRESHOLD`) or sigma is unknown (`UNKNOWN_UNCERTAINTY`). Until a calibration is
published (step 4), tau is the provisional value in `[verification.thresholds]` of
`configs/verifier/checks.toml`, and the summary says `thresholds: provisional`. Ground truth has
no error, so the gravity error of 0.2 against tau 0.15 fails R5b, the energy gain of 0.5 against
0.05 fails R3b, and R5a passes, since the weaker gravity is the same for every flight. A check
can also abstain before the rule, with its reason in `abstain_reason` (the list is
`AbstainReason` in `src/physverify/schema/checks.py`): too few frames around an event
(`TOO_FEW_FRAMES`), an object outside the check's scope (`OUT_OF_SCOPE`, for example a non-sphere
for R3a), a missing input such as a lost track (`MISSING_INPUT`), an occlusion, an
ill-conditioned fit, a scale taken from the fall itself, and a few more. The clip's row combines
its subjects: FAIL if any fails, PASS if all pass, otherwise ABSTAIN with the most common reason,
and the magnitude and sigma are those of the decisive subject. In the R3a row above, `ball_1`
bounces with e = 0.736, so e - 1 = -0.264 is well below tau and it passes, while `ball_0` has
only two rows after its impact and abstains, so the clip abstains on R3a. `status` says whether
any subject was scored (`scored`), all abstained (`abstained`), the check found nothing to judge
(`no_event`, as R2 without a resting body) or it raised an error (`error`).

The nine measures, in their units: R1 is the deepest interpenetration, between two objects or an
object and the ground, over the smaller object's size (`pen_over_size`); R2 is the smallest force
and torque imbalance of the resting bodies over their total weight (`residual_over_weight`); R3a
is the restitution e - 1 of each impact of a sphere with the environment (`e_minus_1`) and R3b
the relative gain of translational kinetic energy in a two-body collision (`dE_over_E`). R4_size
is the largest change of an object's apparent size relative to the other objects in the same
frame (`rel_size_change`), R4_jump the largest third difference of a trajectory over its noise
(`jump_over_noise`, a test statistic) and R4_missing the time an object is inside the image but
unseen without an occlusion to explain it (`seconds`). R5a is the disagreement between
accelerations that should be one gravity (two parts of a flight, consecutive flights of an
object, the first flights of two objects; `accel_rel_diff`, the largest |a_i - a_j| over the
mean |a| of the pair, so at most 2, reached when the two point opposite ways) and R5b the
relative error of a sphere's free-fall acceleration against g, with a scale that does not come
from the fall (`g_rel_error`). Under `--profile precise`, variants replace R3a, R3b, R4_size,
R4_jump and R5a (`[check_variants.profiles.precise]` in `configs/verifier/checks.toml`), and R3b
and R5a then report other units.

With perceived states, the same clips go through detection, tracking, depth and the rest of the
pipeline. `extract fetch-weights` downloads and verifies the weights of the profile (about
880 MB for `fast`, 329 s), and `extract run` fills the perception cache (168 s for the four
clips):

```sh
uv run --no-sync physverify --profile fast extract fetch-weights
uv run --no-sync physverify --profile fast --device cpu extract run --dataset kubric_synthetic \
    --subset local --clips f1_projectile-s000__none \
    --clips f1_projectile-s000__r5b_gravity_scale_L3 --clips f3_collision-s000__none \
    --clips f3_collision-s000__r3_energy_gain_L3
```

```text
detect/grounding_dino_tiny hf:IDEA-Research/grounding-dino-tiny@a2bb814dd30d verified 689.4 MB ...
track/edgetam hf:yonigozlan/EdgeTAM-hf@c266ce53b3fc verified 55.9 MB ...
...
detect/grounding_dino_tiny: cached 0 run 4 total 22.32 s peak rss 2372 MB
track/edgetam: cached 0 run 4 total 102.15 s peak rss 1852 MB
depth/vda_small: cached 0 run 4 total 36.37 s peak rss 1930 MB
camera/assumed_fov: cached 0 run 4 total 0.00 s peak rss 1552 MB
lift/backproject: cached 0 run 4 total 0.19 s peak rss 1554 MB
scale/size_prior: cached 0 run 4 total 0.00 s peak rss 1554 MB
kinematics/savgol: cached 0 run 4 total 0.03 s peak rss 1555 MB
contacts/proximity: cached 0 run 4 total 0.01 s peak rss 1555 MB
```

`verify --level perceived` reads that cache; a detection, tracking or depth stage missing from it
is refused, not run. Most perceived sigmas come from the error profile, the measured error of
each stage. Without one, every command that reads perceived states warns

```text
WARNING physverify.extraction.pipeline: no error profile is loaded for profile fast (paths.error_profile is missing); every measured sigma is unknown (run `diagnose stages --write-profile`)
```

and R1, R4_size, R5a and R5b abstain with `UNKNOWN_UNCERTAINTY` wherever they score.
`diagnose stages` measures the error of each stage against ground truth on calibration clips and
writes the profile (122 s):

```sh
uv run --no-sync physverify --profile fast --device cpu diagnose stages --dataset kubric_synthetic \
    --subset local --write-profile --clips f1_projectile-s000__none \
    --clips f1_projectile-s000__r5b_gravity_scale_L3 --clips f3_collision-s000__none \
    --clips f3_collision-s000__r3_energy_gain_L3
uv run --no-sync physverify --profile fast --device cpu verify --dataset kubric_synthetic \
    --subset local --level perceived --clips f1_projectile-s000__none \
    --clips f1_projectile-s000__r5b_gravity_scale_L3 --clips f3_collision-s000__none \
    --clips f3_collision-s000__r3_energy_gain_L3
```

```text
report: .../data/runs/<run id>/diagnosis/stages.md
error profile: .../data/calibration/stage_errors.fast.json
metric rows: 36
```

```text
run: .../data/runs/<run id>
thresholds: provisional
R1: scored 3 abstained 1 no_event 0 error 0 FAIL 0
R2: scored 0 abstained 0 no_event 4 error 0 FAIL 0
R3a: scored 0 abstained 0 no_event 4 error 0 FAIL 0
R3b: scored 0 abstained 1 no_event 3 error 0 FAIL 0
R4_missing: scored 4 abstained 0 no_event 0 error 0 FAIL 0
R4_size: scored 2 abstained 2 no_event 0 error 0 FAIL 0
R4_jump: scored 4 abstained 0 no_event 0 error 0 FAIL 1
R5a: scored 4 abstained 0 no_event 0 error 0 FAIL 4
R5b: scored 4 abstained 0 no_event 0 error 0 FAIL 1
```

```text
f1_projectile-s000__r5b_gravity_scale_L3,R4_jump,scored,4.856443646097504,1.0,ABSTAIN,INTERVAL_CROSSES_THRESHOLD,3,3,jump_over_noise,1
f1_projectile-s000__r5b_gravity_scale_L3,R5a,scored,1.9989199249840013,0.23167774309689054,FAIL,,6,1,accel_rel_diff,1
f1_projectile-s000__r5b_gravity_scale_L3,R5b,scored,0.3259849045197595,0.7371044584551816,ABSTAIN,INTERVAL_CROSSES_THRESHOLD,1,1,g_rel_error,1
```

Every perceived magnitude is an estimate, and on these clips it shows. On the gravity clip R5b
reads 0.33 +/- 0.74: the interval contains tau, so it abstains instead of claiming the fault.
R4_missing reads 0 s on every clip and still abstains, because at 24 fps its sigma is 0.059 s and
0 + 2 x 0.059 is above tau, two frames (0.083 s). R5a fails on all four clips, the two clean ones
included, with magnitudes from 1.5 to 2.0, and the clean projectile also fails R4_jump (11.1) and
R5b (0.88 +/- 0.17). These are false alarms. For R4_jump and R5b the thresholds are the
provisional hand-set values and an error profile measured on four clips is a rough one; R5b
also falls short of the calibration clause on perceived states, so a fitted tau need not clear
its alarm. The perceived states also differ from the scenes: three objects on the projectile
clips, which have two balls, and neither the impacts with the ground that R3a judges nor the
collision of the collision clips. Step 4 fits the thresholds and the sigma correction on the
calibration split.

R5a's alarms have a cause that no threshold fixes, and on perceived states R5a does not meet the
calibration clause (it is not among the checks counted as met). Its ratio is at most 2, and
perceived depth drives it there. The perceived depth of a small object tends to move against its
true depth: on the perceived states of the calibration split (the `precise` chain), the acceleration
fitted along the line of sight has the wrong sign in 87% of the 858 long flights, and two windows of
one flight can point opposite ways; a window can also span an impact the perceived contacts missed.
On those states the ratio is above 0.15 on all 74 clean clips with an R5a result, against none of
the 55 with ground truth, and its per-clip maximum has a median of 1.87 on the clean clips and 1.90
on the clips with a fault injected for R5a. `precise`, the profile the reported results use, scores
R5a with a Wald contrast of the two accelerations weighted by their propagated sigma
(`accel_wald_chi2_3`, provisional tau 11.34), which has no such bound. On perceived states it does
not separate those faults either: its stratified AUROC on the calibration split is 0.51 [0.41,
0.61], so it does not meet the clause. Selecting it under `fast` with
`--set check_variants.R5a=whitened`, `--set verification.thresholds.R5a=11.34` and
`--set verification.threshold_units.R5a=accel_wald_chi2_3` does not clear these clips: with this
four-clip error profile the clean projectile still fails, at W = 167 +/- 26, and the clean collision
passes.

### 3. Score a real video

The demo runs one video through perception and the checks. Physics-IQ and PISA are the real
datasets of the diagnosis (`[diagnosis.real]`), not sealed sets; `download fetch` gets the `local`
subset of one:

```sh
uv run --no-sync physverify download fetch pisa_real
uv run --no-sync physverify --profile fast --device cpu demo run --prompt "A white bottle falls." \
    --query bottle --video data/raw/pisa_real/0265/movie.mp4
```

```text
pisa_real [local]: 5 clips, 10 files, 88.0 MB, 68.5 s -> .../data/raw/pisa_real
```

```text
adhoc-7f692459__k00 (video-fa91f3d406d1): S = 0.0703 +/- 0.00859, coverage 0.14, rule provisional (provisional)
  .../data/demo/adhoc-7f692459__video-fa91f3d406d1
```

Clip 0265 is 250 frames stored at 30 fps, 1080x1080 (the demo reads the stored rate; PISA's capture
rate is estimated at 120 fps, `real_fps` in `configs/datasets/pisa_real.toml`); the prompt is its
caption in `clip_info.json` and `--query` names the object to detect. The demo copies the video
under `data/raw/local_video/` and writes `report.md`, `report.json` and `overlay.mp4` to the
directory it prints:

```text
> **PROVISIONAL RULE: not a calibrated verdict.**
> The calibrated rule is unavailable: no calibration artifact at .../data/calibration/calibration.json: run `calibrate fit`.
- Rule: provisional, status **provisional**, label `provisional@excl:R4_jump,R5b`
- S = 0.0703 +/- 0.00859; coverage 0.14; 1 checks decided, 1 failed
- Magnitudes from perceived inputs are estimates; sigma is their propagated error ('-' when not yet measured).

| check | verdict | status | magnitude | sigma | threshold | abstain reasons | frames | decided by rule | rule term |
|---|---|---|---|---|---|---|---|---|---|
| R1 | ABSTAIN | scored | 1.15 pen_over_size | - | 0.05 | UNKNOWN_UNCERTAINTY | 17-17 | no | - |
| R2 | ABSTAIN | scored | 0 residual_over_weight | 0.361 | 0.01 | INTERVAL_CROSSES_THRESHOLD, NO_SUPPORT_PATH | 45-51 | no | - |
| R3a | ABSTAIN | abstained | - e_minus_1 | - | - | OUT_OF_SCOPE | - | no | - |
| R3b | ABSTAIN | abstained | - dE_over_E | - | - | TOO_FEW_FRAMES | - | no | - |
| R4_jump | FAIL | scored | 26.7 jump_over_noise | 1 | 3.78 | INTERVAL_CROSSES_THRESHOLD | 20-23 | no | - |
| R4_missing | ABSTAIN | scored | 0 seconds | 0.0471 | 0.0833 | INTERVAL_CROSSES_THRESHOLD | - | no | - |
| R4_size | ABSTAIN | scored | 2.22 rel_size_change | 1.21 | 0.1 | INTERVAL_CROSSES_THRESHOLD | 11-11 | no | - |
| R5a | FAIL | scored | 1.98 accel_rel_diff | 0.261 | 0.15 | - | 134-155 | yes | 0.0703 |
| R5b | NO_EVENT | no_event | - g_rel_error | - | - | - | - | no | - |
```

Each row reads as in step 2: the verdict, the decisive subject's magnitude and sigma, tau, the
reasons of every subject that abstained (so a FAIL can list one) and the frames of the decisive
result. S summarizes the decided checks. Without a usable calibration it is the provisional rule,
s_j = 1 / (1 + m/tau) per decided check and their geometric mean, which is not a calibrated
probability, and the report and the overlay say so. The rule reads seven checks: its label leaves
out R4_jump and R5b, time-dependent checks that `[finetune] exclude_time_dependent` drops
(`configs/verifier/check_scope.toml` lists them), and only R5a is decided, hence coverage 0.14,
so S rests on R5a alone. The video is real footage, so its FAILs are false alarms. R5a reads 1.98
here, near its bound of 2, like the clean projectile of step 2 (2.00), so this S says nothing
about the video; with the `precise` R5a selected as in step 2, R5a abstains (W = 11.5 +/- 7.9
against 11.34) and so does S (`no_decided_check`). The overlay draws S and the rule in a top bar
and one badge per check: green for PASS, red for FAIL (bright inside the frames of the failing
result, dim outside), grey for an abstention and dark grey for no event.

The first run took 783 s, about 600 s of it in tracking, which peaked at about 8.3 GiB of memory
(more than the laptop above has); with the perception cached, a run takes about 90 s. Run the demo
after step 2: without the error profile most sigmas are unknown, no check the rule reads is decided,
and it prints `S = abstained (no_decided_check)`. Its per-check scores are kept under
`data/finetune/demo_adhoc/scores/video-<id>/` and reused for the same video and thresholds even when
the error profile has changed since; move that directory aside to score again. A 4K Physics-IQ
center-view clip from `download fetch physics_iq` (80 frames at 16 fps, 3840x2160) gets its report
but no overlay: decoding it at full size for the overlay needs 1.99 GB, more than
`runtime.max_frame_bytes` (1.50 GB), and the command exits 2.

### 4. Calibration, sealed sets and the final test

```sh
uv run --no-sync physverify --profile fast --device cpu calibrate fit --dataset kubric_synthetic \
    --subset local --level both --clips f1_projectile-s000__none \
    --clips f1_projectile-s000__r5b_gravity_scale_L3 --clips f3_collision-s000__none \
    --clips f3_collision-s000__r3_energy_gain_L3
uv run --no-sync physverify --profile fast calibrate show
uv run --no-sync physverify --profile fast calibrate publish --level oracle
```

```text
artifact: .../data/calibration/calibration.json (insufficient, sha256 433116513fc5)
calibration <run id> v1.10.0 insufficient
oracle/fast: s* -
  R1 insufficient tau 1e-05 pen_over_size clean 2 FPR cv 0 [0, 0.8419] w 0
  R2 no_negatives tau - residual_over_weight clean 0 FPR cv - [-, -] w 0
  ...
perceived/fast: s* -
  ...
  R5a insufficient tau 1.999 accel_rel_diff clean 2 FPR cv 0 [0, 0.8419] w 0
error: calibration.json#oracle/fast@<run id>: the aggregate threshold s* is insufficient: refit on more scenes or pass --force
```

`calibrate fit` fits, per input level, each check's tau on the clean calibration scenes, its
false-positive rate and weight, and for perceived states the bias and sigma correction;
`calibrate show` prints the artifact and `calibrate publish` writes the thresholds `verify` reads.
Two clean scenes are too few, so every check here is `insufficient` (or `no_negatives`, without a
clean event) and `publish` exits 2. Without `--clips`, `fit` reads every calibration clip of
`calibration.subset` (`local` under `fast`, `full` otherwise); on the H200
`scripts/h200/run_perception_iteration.sh` runs the whole chain. A command computes on a set of
`configs/verifier/sealed.toml` only when `PHYSVERIFY_UNSEAL` names it and `src/` and `configs/` are
clean, and it logs the look in the ledger before computing anything. The final test is one command
on the H200, from a clean checkout at the frozen commit:
`bash scripts/h200/final_run.sh [FROM [TO]]` runs stages 0 to 10, from the preflight to
`final_table.{md,json}` and then to the comparison with a VLM judge (`comparison_table.{md,json}`),
with the variables its header lists (`PV_DEVICE` defaults to `cuda`; `PHYSVERIFY_FINAL_DRYRUN=1`
rehearses it on development data). It spends the sealed looks, so it is not run here; see [the H200
README](scripts/h200/README.md) (Sealed sets; Freeze and final looks).

## Development

```sh
uv run --no-sync pytest                 # unit tests (strict markers, config, xfail); no network, GPU or Docker
uv run --no-sync pytest -m docker       # Python 3.9 container boundary (needs the Kubric image)
uv run --no-sync ruff check . && uv run --no-sync ruff format --check .
uv run --no-sync mypy
```

`docker` is the only marker. The code lives in `src/physverify/`.

Data, model weights and run outputs live under `data/` (ignored by git).
