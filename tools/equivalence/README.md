# Behavior-equivalence harness

```sh
bash tools/equivalence/run_equivalence.sh 9eb5888 refactor/architecture       # ~3 min on the laptop, CPU only
Repo/.venv/bin/python tools/equivalence/ast_equal.py <repo> <old-rev> <new-rev> [--renames r.tsv] [--allow a.tsv]
```

`run_equivalence.sh <left> <right>` checks out each revision in a detached worktree at the same
path (`wt/side`), runs the same CLI scenarios with the same absolute run directory (`run/`, so
`config_hash` and every printed path agree), moves the result to `out/<left|right>`, normalizes it to
`norm/<left|right>` and runs `diff -r` (`diff.txt`). It also compares the golden blob list
(`git ls-tree -r <rev> -- tests | awk '$4 ~ /golden\/|synth\/data\//{print $3}' | sort`,
`golden.txt`). Exit 0 only when both are identical and every scenario met its expectation.

Derived inputs are built by each side with its own checkout (`driver.py prepare`, after the
worktree is checked out): the results chain of `tests/reporting/reporting_cases.build_chain` and
the raw oracle run `repo/data/runs/raw-oracle` of
`tests/calibration/calibration_cases.synthetic_report(40)`, so a change that alters how the raw run
is produced shows up in `calibrate fit/show`.
Each side gets its own `configs/` (without `local*.toml`) plus a `local.toml` with the heavy lock
under `run/`. Every scenario runs in a fresh process under `env -i` with
`PYTHONPATH=<wt>/src:<wt>/tests/fixtures:<wt>/tests/calibration:<wt>/tests/reporting:<wt>/tests/checks`; `driver.py`
asserts that `physverify` was imported from `<wt>/src`, adds the `synthetic_fixture`,
calibration fixture and `scenario` (scenes) datasets to `PLUGIN_KINDS["datasets"]` (in `physverify.app.plugins` and/or
`physverify.cli.context`, whichever exist) and calls `physverify.cli.main.run(argv, environ)`.
Old command and option names are translated when the checkout only has the new ones
(`COMMAND_RENAMES`, `OPTION_RENAMES` in `driver.py`). The `help` scenario sorts and labels its
pages by the old command names, so a renamed command's page is compared with its old page.

## Expected differences

Intended behavior changes are recorded in `expected/`, one `.diff` file per change, cumulative from
`9eb5888`: each holds the lines that change adds to `diff.txt` (the output of `diff -r left right`
run inside `norm/`, so paths are relative). The harness sorts `diff.txt` and the concatenation of
the selected files and passes only when they are equal line for line; any extra or missing line is
written to `unexpected.txt`. By default every `expected/*.diff` is selected; `EQ_EXPECTED="a.diff
b.diff"` selects only those names, and `EQ_EXPECTED=` (empty) expects identical outputs, as for
`9eb5888 9eb5888`. To add one, run the harness, check that `unexpected.txt` holds only the new
change, and save those lines (without the `<`/`>` prefix of `unexpected.txt`, i.e. the matching
lines of `diff.txt`) as a new file.

- `wp1-audit-migration.diff`: WP1 (`2b34c0e`) removed `dataset audit-migration`, so its `--help`
  page and its row in `dataset --help` are gone.
  Its hunk headers are where `diff` aligns the deletion, so they move when other help lines on the
  page change (same deleted lines). Since WP9 the row's deletion shares a hunk with a reworded
  line just above it, and that hunk's header (`311,314c312`) is in `wp9-c-help-text.diff`.
- `wp7-config-layout.diff`: WP7 moved every config file into `configs/verifier/`,
  `configs/datasets/` and `configs/studies/` (same tables, same `config_hash`). The flat files are
  `Only in left/repo/configs`, the subdirectories `Only in right/repo/configs`; `doctor`'s
  `source:` lines name the new paths; the `size fit` help names `configs/verifier/size_model.toml`.
  An expected diff rather than a normalization: mapping old paths to new ones in the normalizer
  would hide a file that moved to the wrong place. It retired `wp4a-config-comments.diff` (WP4a's
  renamed-module comments in `configs/checks.toml`, `dataset.toml` and `exclusion_rules.toml`):
  with no file of the same path on both sides, `diff -r` no longer compares those files' lines.
  Coverage lost: from `9eb5888` the harness no longer compares config contents path to path (a
  comment or blank-line change in a moved file passes unseen). A changed table or value still
  changes `config_hash`, which `doctor` prints on both sides, and
  `tests/contracts/test_config_splits.py` pins the split files against the originals.
- `wp8-cli-renames.diff`: WP8's command, option and script renames: the help pages, the stderr
  hints, and the missing-input lines of `results_chain_without_agreement` (stdout, `results.json`
  and `results.md` name `--agreement-run`).
- `wp9-b-messages.diff`: the human-agreement `summary.md` wording (no script reads that file).
- `wp9-c-help-text.diff`: help text (command docstrings and `help=` reach Typer's output) and the
  `configs/default.toml` header comment.

A hunk header lives in the file that holds the hunk's lines. A help line changed by more than one
step belongs to the latest file only, so a hunk with such a line, header included, moves to that
file (WP8's `1093c1073` is now `1093c1076` in `wp9-c-help-text.diff`). A hunk whose lines only
one older step explains stays in the older file, and when a later step shifts the page only its
header numbers change there (WP8's `175c175` became `175c176`). The harness compares the sorted
lines of all the files together, so where a header sits does not change the check.

## Scenarios (each asserts its exit code and that it wrote files or printed output)

The commands and options below are the left side's (`9eb5888`) on purpose: `driver.py` writes each
scenario with those literals and `_map_argv` maps them onto the renamed ones (`COMMAND_RENAMES`,
`OPTION_RENAMES`) only on a checkout that lacks the old ones.

| Scenario | Command | Exit |
|---|---|---|
| doctor, doctor_set | `doctor`; `--set runtime.threads=2 --profile precise doctor` | 0 |
| verify_oracle, verify_oracle_sweep | `verify --dataset synthetic_fixture --level oracle [--sweep]` | 0 |
| verify_perceived_raw | `verify ... --level perceived --policy raw` | 0 |
| verify_scenes_oracle, verify_scenes_raw, verify_scenes_sweep | `verify --dataset scenario --level oracle [--policy raw \| --sweep]` | 0 |
| verify_usage_error, verify_unknown_dataset | no `--level`; `--dataset nope` | 2 |
| dataset_generate_smoke, dataset_generate_again | `dataset generate --subset smoke` twice | 0 |
| dataset_list_smoke, dataset_list_local | `dataset list --subset smoke`; unplanned `local` | 0; 2 |
| results_chain, results_chain_without_agreement | `results` on the chain directory [and the agreement run], under the left side's option names | 0 |
| calibrate_fit, calibrate_show | `calibrate fit --from-run raw-oracle --out ...` (bootstrap 100), `show` | 0 |
| calibrate_publish_refused | `publish` of a non-calibrated artifact with `require_calibrated_artifact` | 2 |
| root_usage_error | unknown subcommand | 2 |
| help | `--help` of the root, every command and every subcommand (88 pages) | 0 |

Skipped, and why: `extract` with real providers, `diagnose`, `timing`, `demo`, `finetune`, `h3` and
`dataset simulate/render` need torch weights, Docker/udocker or a GPU. `calibrate evaluate` and
`agreement compare` are covered by their tests only through monkeypatched internals
(`calibrate._raw_runner`, the agreement `world` fixture) that the refactor moves, so a scenario
would test the harness, not the CLI.

The `scenario` dataset (`ScenesDataset` in `driver.py`) serves 14 analytic clips of
`tests/checks/check_scenarios.py` (`SCENES`: contact, sinking and floating boxes, an overhang,
bounces with and without sigma, a gaining collision, a disappearance, a resized pair, a size change
with sigma, teleports, a gravity change and a strong-gravity parabola) as oracle ground truth in the
camera frame. Every check R1, R2, R3a, R3b, R4_jump, R4_missing, R4_size, R5a and R5b gives a
non-zero magnitude on at least one clip, so a change to any check's number reaches `checks.jsonl`,
`table.csv` and, through the sweep, `specificity.json` and `sweep_matrix.csv`. The `synthetic_fixture`
clip alone gave magnitudes of 0 or about 1e-14 and R3a/R3b only abstained.

## Normalization (only these)

Run ids `[0-9]{8}-[0-9]{6}-[0-9a-f]+` become `RUNID<k>` in order of first appearance (dir names
too); ISO timestamps (also `YYYY-MM-DD HH:MM:SS,mmm` log stamps) become `TS`; both revisions' commit
hashes (full and 12-char) become `COMMIT`; the run directory becomes `RUN` and the worktree
`WORKTREE`; a sha256 (full or 12-char prefix) of a file the run wrote becomes `SHA(<file>)`, because
artifacts that embed `created_utc` hash differently on every run; `.npz` files are compared by entry
name, size and payload sha256 (zip entry times change on every write).

## Baseline

Outputs go to `$EQ_WORKDIR` (default `$TMPDIR/physverify-equivalence`); rerun `EQ_EXPECTED= ... 9eb5888 9eb5888` to rebuild the base baseline (161 files, scenes scenarios included).
Results recorded with the current scenario list and per-side inputs:

- 9eb5888 vs 9eb5888: identical (161 files); 59 golden blobs identical.
- Sensitivity proof: a throwaway commit ef966d1 on top of 6dc9f86 that multiplies the R1 measure
  (`penetration_ratio`) by 1.000001. 6dc9f86 vs ef966d1 exits 1, "normalized outputs DIFFER (208
  lines)": `checks.jsonl` and `table.csv` of the three scenes scenarios, plus `specificity.json` and
  `sweep_matrix.csv` of the sweep. Before the scenes dataset the same injection was reported
  identical.
- 9eb5888 vs refactor/wp7-k (WP7): differs only as expected (`wp1-audit-migration.diff`,
  `wp7-config-layout.diff`); `config_hash` and every JSON output identical; 59 golden blobs
  identical.
- 9eb5888 vs 0d4b15b (WP1 head plus the audit fixes): only `help` differs, by the removed
  `dataset audit-migration` page and its row in `dataset --help` (expected); every other scenario,
  the scenes ones included, identical; 59 golden blobs identical.

## ast_equal.py

For pure-move and comment/docstring commits: pairs modules by git rename detection (`-M30%`) plus
`--renames` (`<old path>\t<new path>`), strips docstrings, rewrites the old side's imports and
`physverify...` string constants through the module-rename map, applies `--allow` substitutions
(`<new path or *>\t<name|literal|alias|variable>\t<old>\t<new>`; an `alias` line turns the old
`import <old>` into `import <name> as <asname>`, written `<name> as <asname>`; a `variable` line
renames variables and imported names but not attributes, for a module alias that shares its name
with an attribute) and compares `ast.dump`. With `--sort-imports`
(used for the renames of WP4a) each run of consecutive import statements and each import's names
are sorted on both sides first, so the import reordering `ruff` makes after a rename is not a
difference. A string constant follows the module-rename map only when it names a
paired module or a dotted descendant, and never when the old revision assigns that value at module
level to a `*_SCHEMA`, `*_FILE`, `*_VERSION`, `LEDGER_LABEL*` or `GUARDED` name, so a persisted
schema id cannot move with a package rename unseen. `--strip-bare-strings` removes every
string-constant statement on both sides (attribute docstrings too), for comment and docstring
commits. Reports added, deleted and differing modules and every `--allow` line that was never
applied; exit 0 only when there are none. Comments are not in the AST, so comment-only changes
are equal by construction. `tests/equivalence/test_ast_equal.py` pins these cases.

The harness may need adapting when a later step changes `physverify.cli.main.build_app`, the
commands registry location or the fixture modules' locations; adapt `driver.py`, never the
scenario list's meaning.
