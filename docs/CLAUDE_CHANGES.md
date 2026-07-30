# Change log

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
