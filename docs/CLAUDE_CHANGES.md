# Change log

## 2026-08-01 — two new languages, a re-serial, and the volume hypothesis weakens

### wandb hygiene
`en_us`/water had been run twice under serial 0, breaking the one-row-per-cell contract. New
**`rename_wandb_serial.py`** (selection by explicit run id, because a re-run is
config-identical to its original) moved the superseded 2026-07-27 run to **serial 1** and
rewrote its name suffix. Dry-run by default; applied and verified.

### The grid is now 12 languages
`am_et` (Amharic, ~40 h) and `ur_pk` (Urdu, ~80 h) fill the low and middle resource tiers, and
`ta_in` gained `base` + `Qwen3-4B`. `RESOURCE_TIER` encodes the four tiers; all four are now
populated, which was the point of the additions.

**Spanish changed underneath the analysis.** The Spain-Spanish (`es_es`) runs were deleted and
re-run from `es_mx`. That halves the cell's stream (866k -> 206k clips) and removes its
dialect-mismatch flag, since Mexican Spanish is inside the Latin American variety FLEURS
`es_419` evaluates.

**`en_us`/water is no longer excluded.** It replicated (12.05 vs 17.06, both far worse than the
other variants), so it is an effect rather than a failed run -- and since water is English's
matched variant, excluding it had been removing the strongest against-hypothesis point.
`EXCLUDED_MODELS_AGGREGATE` is now empty.

### The headline result weakened, and that is the finding
With 7 languages the volume trend was rho 0.96, p 0.0005, robust to dropping Tamil. With 10 it
is rho 0.721, p 0.0186 -- and **no longer robust** (p 0.0769 without Tamil). The cause is one
cell: `am_et` has 8,873 clips against `ta_in`'s 8,846 but a delta of +0.64 versus -14.70. Same
volume, opposite sign. Tamil's outlier is about Tamil, not about being low-resource. Region
matching itself is a clean null: mean -0.81 CER, p 0.695, MDE 5.42.

### Added
- **`analyze_loss_metrics.py`** (t6): separates overfitting (eval loss rising after its own
  minimum) from domain/accent shift (a large train-eval gap with no rise). Overfitting is
  monotone in resource tier (0.195 / 0.028 / 0.003), and cross-domain evals show a 6x larger
  generalisation gap with the same near-zero rise -- the two are separable.
- **`fetch_tinyaya_composition.py`**: parses the Tiny Aya report's Appendix A tables
  (arXiv:2603.11510, Tables 8-14) into a CSV, turning "specialisation" from a categorical label
  into a continuous per-language exposure variable. Stdlib-only parser (no lxml). Validated by
  reproducing each region's printed Subtotal and by checking the mix->variant mapping against
  the numbers. Caught its own bug: `Subtotal` rows were first counted as languages, inflating
  every mix total to ~180 %.
- **`for_quantizedasr/`**: a WorldSpeech in-domain eval config generator and a both-domains
  eval sweep (serial 420), written in QuantizedASR's style but kept here, since that repo is
  not modified from this session.

### Corrections
- The regional definition was wrong: Earth is Africa **+ West Asia**, Water is Asia-Pacific
  **+ Europe**. No executed language is West Asian so no assignment changed, but the definition
  is now recorded correctly.
- Urdu is ambiguous: the report's Table 1 places it in South Asia, Appendix Table 10 under West
  Asia. Assigned `fire` because the fire mix carries 3.4 % Urdu against 1.3 % earth.

---

## 2026-07-30 (fifth pass) — a filter bug becomes the study's central finding

The strict `< 30 s` duration cap silently discarded 72.4 % of the Tamil training data, and
understanding that changed the paper's claim rather than just one table.

### Root cause

`make_audio_length_filter_fn` keeps a clip when `length < max_input_length` — strict — and every
`configs/train/*ws*.yaml` sets `max_input_length: 30`. WorldSpeech `ta_lk` is pre-segmented into
fixed 30-second windows (100/100 sampled rows at exactly 30.00 s), so every clip fails `30.0 < 30`.
Filtering the interleaved Tamil stream leaves exactly **8,846** rows == `len(ta_in)`, out of an
intended 32,107. **23,261 clips lost, with nothing in the logs reporting it.**

This vindicates `analyze_data_accounting.py`, which inferred 8,825 against a true 8,846 (99.76 %).
Against the post-filter expectation all nine languages now reconcile. Written up for upstream in
the new **`docs/UPSTREAM_FIXES.md`**; QuantizedASR is not modified.

### The mistake I nearly shipped

I planned to **exclude** Tamil as a contaminated cell. That was wrong, and the correction is the
finding. The region-match contrast is computed *within* a language: all four variants consumed the
identical 8,846-clip stream, so the loss reduced every arm equally and cannot bias the comparison.
It only relocates Tamil on the data-volume axis, where it is the grid's only genuinely
low-resource cell — the most informative point, not a corrupt one. `utils.py` now carries an
explicit note against re-introducing the exclusion, and a test asserts `ta_in` is in no exclusion
list.

### The finding

Ordering the seven usable languages by training-stream size, the matched-decoder benefit decays
monotonically from **-14.70 CER at 8,846 utterances to +1.06 at 577,382**: Spearman
**rho = 0.964, p = 0.0005**, surviving removal of the extreme point (**rho = 0.943, p = 0.0048**).

Reported with the confound rather than around it: volume and baseline CER are collinear
(rho = -0.679), the *relative* effect trends only weakly (rho = 0.714, p = 0.0713), and the
partial correlation controlling for difficulty is inconclusive over all seven (r = 0.137, p = 0.77,
df = 4) though significant over the six non-extreme languages (r = 0.838, p = 0.04). The decisive
test is a within-language volume manipulation — re-run Tamil at full volume, keep the low-volume
runs — which is exactly the 4-run budget available.

### Added

- **`analyze_volume_interaction.py`** → `t5_volume_interaction.csv`, `t5_volume_stats.csv`.
- **Two figures**: the decay curve and its relative-effect companion, so the confound is visible.
- **`utils.py`**: `MAX_INPUT_LENGTH_S`, `CONFIG_DURATION_AT_CAP` (frozen at-cap screen),
  `KNOWN_AT_CAP_CONFIGS`, and `expected_stream_examples(post_filter=)`.
- **`verify_dataset_durations.py`**: an at-cap screen for any config. Snapshot-first with the live
  datasets-server sample as cross-check, because the endpoint 500s for uncached configs and a
  screen that degrades to "inconclusive, exit 0" is how this bug survived. Known-bad configs are
  reported as `[KNOWN]` so the guard gates on new regressions instead of failing forever.
- 77 unit checks (up from 79 total across both test scripts), 29 ordering claims, 94 numbers.

### Two bugs found in my own code along the way

- **`plot_curve.get_hue_order` was non-deterministic.** It iterated a `set` for hue levels absent
  from `METHODS_DIC`, and CPython randomises string hashing per process — so region colours
  changed between renders while point positions and labels stayed pixel-identical. That is the
  exact silent-colour-shift defect the repo conventions warn about, reproduced by me. Now sorted;
  verified byte-identical across processes, and confirmed the old path really did vary
  (`PYTHONHASHSEED=3` gives a different set order than 1 or 2).
- **A `--load` path that had never run.** `make_audio_length_filter_fn` copied the upstream dict
  access, but under `datasets` 4.x with torchcodec the `Audio` feature yields an `AudioDecoder`,
  so the bare `except` would have marked every clip corrupt. Fixed and exercised for real.

---

## 2026-07-30 (fourth pass) — the WorldSpeech "malformed files" problem is an env problem

Comparing `pip list` from the remote host that *can* read WorldSpeech against the two local
conda envs identified the cause, and it is not libsndfile or Opus.

| | `pytorch` (local) | `asr` (local) | remote host |
|---|---|---|---|
| `datasets` | 5.0.0 | 4.5.0 | 4.5.0 |
| `transformers` | 5.14.1 | 4.57.5 | 4.57.5 |
| `soundfile` | **absent** | 0.14.0 | 0.13.1 |
| `torchcodec` | **absent** | 0.9.1 | 0.9.1 |
| `torchaudio` | **absent** | absent | 2.9.1+cu130 |
| ffmpeg/libsndfile `.so` | **0** | 18 | — |
| `scipy` | present | **absent** | — |

`pytorch` has **no audio backend whatsoever**: `datasets` 5.0.0 decodes `Audio` through
torchcodec, and neither torchcodec nor soundfile is installed, so *every* audio read fails
there regardless of codec. `asr` matches the remote host closely — including `transformers`
4.57.5, which is exactly the `transformers_version` logged by the runs — so `asr` is the env
that reproduces how training consumed data.

**Retested in `asr`, and the blocker is gone.** The three configs recorded upstream as
undecodable (`examples/explore_datasets.py:30`, "Supported file format but file is malformed")
all decode correctly via torchcodec at 24 kHz: `la_va` 29.54 s, `si_lk` 29.34 s, `tl_ph` 0.80 s,
and `ta_in` 18.01 s for the first test clip of each. The libsndfile error string means an env
without torchcodec fell back to soundfile; it was never a property of the corpus.

Consequences:

- **`CLAUDE.md` corrected.** It stated "There is no `asr` env on this machine, whatever older
  docs say." Both halves were wrong: `asr` exists and it is the only local env that can read
  audio. The file now carries the split — analysis and `plotter.sh` in `pytorch` (it has scipy,
  which `asr` lacks), anything loading a dataset in `asr`.
- **`EVAL_DATASET_PLAN.md`'s top-priority tier is unblocked.** Adding matched WorldSpeech `test`
  eval configs was written up as blocked on a decode bug; it is ordinary work.
- **The interleave guard is version-robust.** It was originally verified on `datasets` 5.0.0;
  re-run under `asr`/`datasets` 4.5.0 — the version the runs actually used — it passes
  identically, which matters because the earlier verification and the runs were on different
  major versions.
- **A real bug in `verify_dataset_durations.py`, found because the env difference forced the
  question.** Its `make_audio_length_fn` was a verbatim copy of the upstream function, which
  reads `audio['array']`. Under `datasets` 4.x with torchcodec the `Audio` feature yields an
  `AudioDecoder` instead, so dict access raises and the bare `except` would have assigned the
  corrupt sentinel to **every** clip — reporting a whole corpus as broken and filtering the
  split away. The `--load` path had never been exercised, so this was live. It now handles both
  shapes, and `--load` has been run for real: `ta_in` + `ta_lk` `test`, 1690 == 466 + 1224,
  0 undecodable, 0 removed by the duration filter, loaded counts matching builder metadata.
  That independently reproduces on `test` what was reported on `train`.

---

## 2026-07-30 (third pass) — the ta_in "anomaly" was mine, not the data's

A direct test on the real datasets refuted the last remaining suspicion about `ta_in`, and the
input that produced it has been replaced.

### What was tested, and what it showed

`disco-eth/WorldSpeech`, configs `ta_in` + `ta_lk`, split `train`, loaded map-style with
`audio_length_s` computed from the decoded arrays:

* `len(interleaved) == len(ta_in) + len(ta_lk)` — interleaving loses nothing on real data.
* The duration-consistency filter (`|audio_length_s − duration| < 1 s`) removed **zero**
  samples — no corrupt clips, no mislabelled durations.

Both mechanisms the previous pass had proposed for a Tamil data problem are therefore false.

### The bad input: hours recovered from prose

The flag came from comparing a reconstructed stream size against per-language hour figures
taken from a **summarised web fetch** of the WorldSpeech paper's table — not from reading the
table. That was the weakest-provenance input in the whole analysis, and it was gating a
finding. It claimed 240 h for `ta_lk`, which cannot be right for 23,261 clips.

Replaced by `utils.WORLDSPEECH_TRAIN_EXAMPLES`: `num_examples` per training config, read from
the HuggingFace dataset builder metadata. Authoritative, instant, no audio downloaded. Also
added `TRAIN_CONFIGS` (the `(path, configs, split)` each language trained on) and
`expected_stream_examples()`, which sums the parts — valid precisely because interleaving is
lossless.

### What the corrected accounting says

Eight of nine languages reconcile, including the other two multi-config languages, whose
reconstructions match their **summed** streams (`ha_ng` 1.04, `sw_ke` 1.20) and not a single
config (1.84, 1.81). `ta_in` is the exception, and its reconstruction matches **one config to
within 0.24 %** (ratio to expected 0.27; to nearest single config 1.00).

That is a bookkeeping question, not a data question: either the run consumed one config, or the
epoch counter is unreliable for it, and the logs do not separate the two. The only consequence
kept in the analysis is that Tamil is the smallest-data cell, has the worst CER, and carries
the largest region-match term (−14.70) — so that term is provisional on **data-volume**
grounds, not data-quality grounds. `t4`'s docstring now states explicitly that it is not an
integrity check.

### New: `verify_dataset_durations.py`

The test above, generalised into a checker for any `--dataset_path --dataset_configs --split`,
so the next suspicion of this kind is tested before it is written down. Two modes: metadata
only (instant, authoritative counts, wired into `plotter.sh`) and `--load` (downloads audio,
asserts the interleave arithmetic on real objects, and reports exactly how many samples the
duration filter removes). The `--load` invocation is recorded in `claude_process.sh`.

### Also caught

`analyze_region_match.py` still referenced a renamed column and `set -e` — added in the
previous pass — aborted the run rather than letting it through. The checker then flagged a
`crs_sc` ratio that is legitimately undefined (its ERISLab mirror split is not in the snapshot)
and is now excluded from that specific check rather than being papered over.

Rebuild after the corrections: exit 0, **79** checks pass, 21 ordering claims unbroken, 85
numbers verified against their CSVs.

---

## 2026-07-30 (later) — three corrections after review

Three claims from the first pass were wrong or misplaced. All are corrected in code, not just
in prose, so they cannot quietly return.

### 1. Interleaving is NOT a confound — refuted empirically

The first pass flagged `ta_in`/`ha_ng`/`sw_ke` as `uniform_interleave`, reasoning from the
loader that uniform `1/N` probabilities would oversample the smaller config. That reading was
wrong. The strategy is `all_exhausted_without_replacement`, a real supported strategy
(`datasets` 5.0.0 lists it in the `interleave_datasets` signature), and it never recycles an
exhausted dataset — so the combined stream is exactly the sum of its parts.

**`verify_interleave_semantics.py`** (new, offline, in `plotter.sh`) proves it with exact
counts: interleaving a 100-example and a 25-example dataset yields exactly 125 distinct
examples, every example once, in both the map-style and streaming paths, and exact for a
three-way pairing. Plain `all_exhausted` with the same uniform probabilities oversamples the
smaller dataset to 122× (total 222) — the behaviour originally assumed. Verified against the
loader at commit `ac7566e`, which is what the live experiments run.

`TRAIN_EVAL_MATCH` no longer carries the flag, `MULTI_CONFIG_TRAIN` records the pairings
without implying oversampling, and a test asserts the flag stays absent.

### 2. The `crs_sc` `_clean` splits are all genuinely cleaned

The first pass claimed `val_clean` was unfiltered because the committed upstream example only
demonstrates the duration-consistency filter on the `test` split. The filter was in fact
applied to train and val too when the splits were built. `analyze_ood_crs.py` now records this
as a note rather than a caveat, and the docs are corrected.

### 3. Hours/epochs analysis moved here, and no longer needs the training framework

The earlier plan proposed a dataset-hours utility inside QuantizedASR. That repo is not to be
touched. **`analyze_data_accounting.py`** (new) reconstructs each training stream from logged
scalars alone — `global_step × batch_size × gradient_accumulation_steps`, audio seconds, and
the epoch counter, which under streaming counts stream consumptions — and compares it against
a **frozen WorldSpeech hour snapshot** in `utils.WORLDSPEECH_HOURS` (arXiv:2605.09167,
*since removed -- replaced by `WORLDSPEECH_TRAIN_EXAMPLES`, see the third-pass entry*;
copied as objective raw data, never referenced live).

Each entry carries a `scope`, which is load-bearing: only four languages support a direct
config-to-config comparison, three are lower bounds (a second config's hours are unknown), and
three (`en_us`, `fr_fr`, `es_419`) have only a language-level aggregate over configs the runs
never used. Comparing without reading the scope manufactures discrepancies.

Result: **`ta_in` is the sole anomaly** — implied stream 34.82 h against a lower bound of
240 h, ratio 0.15, so the true shortfall exceeds 6.7×.
*(SUPERSEDED by the third pass: the 240 h figure came from a summarised reading of the paper
and was wrong. Direct testing shows the Tamil data is clean. See the entry above.)* Every other comparable language
reconciles (`ha_ng` 1.00, `crs_sc` 1.03, `sw_ke` 1.04, `id_id` 1.28; `mr_in` 2.07 is above
published, which is not a data-loss failure). With interleaving ruled out, the remaining
candidate is the silent filter path: clips of 30 s or more dropped, and undecodable clips
removed via the 100,000 s sentinel without being counted. `ta_in` received no cleaning pass.
This matters because `ta_in` carries the largest region-match term (−14.70 CER).

### 4. `plotter.sh` was not actually gating on its guards

Found while fixing the above: the script had `set -u` but not `set -e`, so its exit status was
only the last command's. A failing `test_utils_port.py` printed `[FAIL]`, exited 1, and
`plotter.sh` still reported success. Now `set -eu`, and verified by deliberately breaking a
check: the run aborts with exit 1 before reaching `verify_paper_numbers.py`.

### Verification after the corrections

`bash plotter.sh` from an emptied `results_all/`: exit 0, **77** unit/semantic checks pass with
0 failures, 19 ordering claims unbroken, 75 numbers verified against their CSVs, 4 figures
regenerated. The checker again caught a drifted number while these edits were made (`id_id`
epochs 5.015 printed as 5.01; correct is 5.02) and a stale ordering claim that assumed
`es_419` appears in the by-language table when it has no finished runs.

---

## 2026-07-30 — first data pipeline

The repo went from seeded-but-empty to a working pipeline over wandb `LisTAya/LALMDecoder`
serial 0 (41 runs). `bash plotter.sh` now rebuilds everything from a bare checkout.

### Added

- **`download_wandb_history.py`** — the time-series download, gated behind `--history`. One
  row per (run, step) via `run.scan_history()`. This is the first history-aware tooling in
  the `/home/edwinrios/analysis/` family; nothing else touches wandb history. ~68 s and
  ~7.5k rows for serial 0.
- **`plot_curve.py`** — training-curve plotting. Separate from `plot.py` because that
  script's `line` branch force-melts onto a hardcoded metric list, `keep_columns` drops the
  needed columns, and `rename_vars` rewrites the axis arguments; editing it would change
  every existing sibling-repo figure. Reuses the same theme, style arguments and save
  convention. Adds faceting, an **explicit** `--errorbar` (seaborn's silent default is
  `('ci', 95)`), the positive-value log-axis fix, and `mkdir -p` on the output subdirectory.
- **`analyze_sample_efficiency.py`** → `t1_sample_efficiency.csv`
- **`analyze_region_match.py`** → `t2_region_match.csv`, `t2_region_match_stats.csv`
- **`analyze_ood_crs.py`** → `t3_crs_ood.csv`
- **`test_utils_port.py`** (52 checks) and **`verify_paper_numbers.py`** (12 ordering claims,
  46 numbers), both wired into `plotter.sh`. Built alongside the first table rather than
  retrofitted.
- **`plotter.sh`**, **`claude_process.sh`**, `NUMBER_PROVENANCE.md`,
  `REPRODUCE_FIGURES.md`, `docs/PLAN_ASSESSMENT.md`, `docs/EVAL_DATASET_PLAN.md`.

### Changed

- **`utils.py`** — placeholder dicts populated from the real runs. The seeded `METHODS_DIC`
  keys (`q2a_whisper_small_tiny_aya_*`) matched **nothing**: the runs use whisper-**medium**
  and full HF paths. Added `MODEL_SHORT`, `CORE_VARIANTS`, `LANGUAGE_REGION`,
  `LANGUAGE_STATUS`, `LANGUAGE_DIC`, `TRAIN_EVAL_MATCH`, and the helpers `half_up`,
  `assert_unique_keys`, `add_language_columns`. `EXCLUDED_MODELS_AGGREGATE` now records the
  failed `en_us`/`water` run with its reason.
- No changes to `download_save_wandb_data.py`, `concat_df.py`, `count_data.py`, `plot.py` —
  their defaults are all CLI-overridable, so `plotter.sh` passes the right key set per run
  type.

### Findings worth not rediscovering

- **The seeded `SUMMARY_COLS` return nothing for training runs.** Real keys are `eval/cer`,
  `eval/loss`, `train/train_audio_seconds`; `no_params` is `model/num_parameters` and lives
  in *config*. The `wer`/`cer`/`rtfx`/`no_params` columns are kept because planned eval-only
  runs will populate exactly those.
- **Effective batch is 512** (`batch_size` 8 × `gradient_accumulation_steps` 64). Missing the
  accumulation factor makes seconds-of-audio-per-sample come out at ~956 s against a 30 s
  cap — impossible, and the sign that the factor was dropped. `test_utils_port.py` pins it.
- **Unique hours and epochs are not derivable from wandb here.** `train/train_audio_seconds`
  is a real measurement, but `train/epoch` is an estimate under `streaming=True`.
  *(Partly superseded by the later entry: `t4_data_accounting` does reconstruct stream hours
  from the epoch counter, but always tagged `estimate` or `lower_bound`, and compared only
  against published figures whose `scope` permits it. Curves still use audio processed.)*
- **The endpoint comparison is underpowered.** Median within-run late-training CER sd is 1.03
  against a mean region effect of 0.48; minimum detectable effect is 6.97–9.27 CER.
- **Confounds encoded in `TRAIN_EVAL_MATCH`**: `fr_fr`/`es_419` are cross-dialect
  (trained `fr_ca`/`es_es`, evaluated `fr_fr`/`es_419`).
  *(SUPERSEDED in part: this entry also claimed `ta_in`/`ha_ng`/`sw_ke` were confounded by
  50/50 interleaving. That was wrong — see the later entry and
  `verify_interleave_semantics.py`. The flag has been removed.)*
- ~~**`crs_sc`'s `val_clean` split is not duration-filtered** despite the name.~~
  *(WRONG, corrected in the later entry: train, val and test all carry the cleaning.)*

### Verification

- `bash plotter.sh` from an emptied `results_all/`: exit 0, 52 unit checks pass, 12 ordering
  claims unbroken, 46 numbers verified, 4 figures regenerated.
- Both checkers negative-tested (`claude_process.sh`, spent lines commented out). The first
  attempt perturbed t1 row 0 and the checker stayed green — correctly, since row 0 is
  `crs_sc`/`base` and every checked aggregate is over `CORE_VARIANTS`. Retargeted; all three
  perturbations then went red and restored green.
- The checker caught three drifted numbers in `PLAN_ASSESSMENT.md` on first run, including a
  double-rounding defect (`0.644882` written as `0.65`; correct is `0.64`) — the exact class
  of bug the conventions warn about.
