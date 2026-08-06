# Generated files: what exists, what writes it, and how to rebuild

Merges the number-provenance and figure-reproduction notes. Every number in `FINDINGS.md`
traces to a CSV below, and `verify_paper_numbers.py` re-derives all of them on every run.
Adding a number to a document means adding it to that checker's `DERIVED` list.

Nothing is hand-computed. Statistics are stored at 6 dp so a printed value is rounded exactly
once, and printing uses round-half-**up**, not Python's banker's default.

## Rebuild everything

    conda activate pytorch
    bash plotter.sh

Downloads are guarded on file existence, so re-running does not re-download; delete the file
under `data/raw_serials/` to force a refresh. Serials 10 and 11 are deliberately unguarded
because they fill in incrementally.

Anything that loads a dataset needs `conda activate asr` instead — `pytorch` has no audio
backend at all.

## Downloaded data

| File | Written by | Contents |
|---|---|---|
| `data/raw_serials/raw_serial_{0..5}.csv` | `download_save_wandb_data.py` | one row per training run: config + summary scalars |
| `data/raw_serials/history_serial_{0..5}.csv` | `download_wandb_history.py --history` | one row per (run, step): the full training/eval curves |
| `data/raw_serials/raw_serial_{10,11}.csv` | `download_save_wandb_data.py` | eval-only runs: 10 = off-the-shelf baselines, 11 = trained checkpoints |
| `data/dataset_checks/*.csv` | `verify_dataset_durations.py` | per-config example counts and at-cap fractions; with `--load`, duration consistency and interleave integrity |
| `data/tinyaya_report/tinyaya_language_composition*.csv` | `fetch_tinyaya_composition.py` | per-language share of each variant's post-training mix |

## What each serial holds

A serial means one thing, so an aggregate over serial 0 is correct without further filtering.

| serial | n | contents |
|---|---|---|
| **0** | 48 | the grid: 12 languages x {earth, fire, global, water}. **The analysis population.** |
| 1 | 8 | superseded grid re-runs (`am_et`, `crs_sc`, seed 42, replaced by seed 420) |
| 2 | 4 | control arms: `base` and `qwen3-4b`, on `crs_sc` and `ta_in` only |
| 3 | 1 | superseded control (`crs_sc`/`base`) |
| 4 | 1 | same-seed replicate (`en_us`/`water`) |
| 5 | 4 | superseded condition: trained on `es_es`, replaced by `es_mx` |

t1 spans **serials 0 and 2** so the `crs_sc` cell keeps all six models; `serial` is a column and
every cross-language table filters to 0. Replicate pairing runs over 0<->1, 2<->3 and 0<->4.
Serials 3, 4 and 5 exist to retain the runs, not to be analysed in depth.

## Analysis tables

All in `results_all/acc/`.

| File | Written by | Contents |
|---|---|---|
| `count_serial_0.csv` | `count_data.py` | per-dataset run counts and state breakdown |
| `t1_sample_efficiency.csv` | `analyze_sample_efficiency.py` | per-run curve statistics; `is_canonical` names the run each cell reports |
| `t2_region_match.csv` | `analyze_region_match.py` | per-language matched / mismatched / global |
| `t2_region_match_stats.csv` | `analyze_region_match.py` | paired tests, bootstrap CIs, minimum detectable effect |
| `t3_crs_ood.csv` | `analyze_ood_crs.py` | the `crs_sc` out-of-distribution cell, all six models |
| `t4_data_accounting.csv` | `analyze_data_accounting.py` | per-run stream reconstruction |
| `t4_data_accounting_by_language.csv` | `analyze_data_accounting.py` | per-language stream vs authoritative example counts |
| `t5_volume_interaction.csv` | `analyze_volume_interaction.py` | per-language effect size against training-stream size |
| `t5_volume_stats.csv` | `analyze_volume_interaction.py` | correlations, drop-one robustness, collinearity, partial correlation |
| `t6_loss_metrics.csv` | `analyze_loss_metrics.py` | per-run generalisation gap and eval-loss rise |
| `t6_loss_by_axis.csv` | `analyze_loss_metrics.py` | the same, grouped by domain / accent / resource tier |
| `t7_baselines.csv` | `analyze_baselines.py` | serial 10 baselines (whisper-medium + Voxtral-Mini + Qwen2-Audio x 44 datasets), with `eval_domain` and `in_domain_role` |
| `t7_training_vs_baseline.csv` | `analyze_baselines.py` | serial 11 minus serial 10; written only once serial 11 exists |
| `t8_exposure.csv` | `analyze_exposure.py` | per-language treatment size in percentage points |
| `t8_exposure_stats.csv` | `analyze_exposure.py` | exposure-vs-effect correlations |
| `t9_replicates.csv` | `analyze_replicates.py` | serial 0 against serial 1, per replicate pair |
| `t9_replicate_stats.csv` | `analyze_replicates.py` | pooled between-run standard deviation |

## Missing-run checks

`missing_runs.py` diffs a downloaded serial against the grid that should exist and writes
`results_all/<serial>_missing.sh` with the commands to fill the gaps. A cell that failed once and
later succeeded counts as done; MISSING, FAILED and CRASHED all rerun. `--model_dir` must point at
a QuantizedASR checkout for the model YAMLs.

The serial-10 grid is an explicit `MODEL_CONFIGS` x `DATASET_CONFIGS` cross product held in the
script; `test_utils_port.py` asserts both lists match `eval_lalm_baselines.sh` so the duplication
cannot rot. Serial 11 is not a cross product — each checkpoint pairs only with its own language —
so it needs `--pairings <sweep>`.

## Figures

All written by `plot_curve.py` into `results_all/plots/s0/`.

| Figure | Content |
|---|---|
| `s0_curve_cer_vs_audio_hours.png` | Eval CER vs audio processed, hue = decoder variant, faceted by language |
| `s0_curve_evalloss_vs_audio_hours.png` | Eval loss on the same axes — separates "still learning" from "CER is noisy" |
| `s0_curve_trainloss_vs_audio_hours.png` | Training loss on the same axes |
| `s0_curve_crs_ood.png` | The `crs_sc` cell alone: unseen by both encoder and decoder |
| `s0_volume_interaction.png` | **Headline.** Region-match effect against training-stream size, one point per language |
| `s0_volume_interaction_relative.png` | The same effect as a share of baseline CER, so the volume/difficulty confound is visible |

## Definitions a reader could otherwise get wrong

- **`best_cer`** is the minimum CER over a run's 101 evaluations, and is the **primary** metric.
  `final_cer` is the last evaluation and is secondary: mean `final_minus_best` is 5.36, so the
  last checkpoint is frequently not the best.
- **Aggregation population.** Every aggregate in `FINDINGS.md` is over **finished, canonical
  runs of the four grid-wide variants** (`earth, fire, global, water`). `base` and `Qwen3-4B`
  exist only for `crs_sc`, so including them would compare a 12-language mean against a
  1-language mean.
- **`is_canonical`.** During a re-run window a cell holds two runs in the same serial. t1 names
  one as canonical — finished first, then earliest — and every downstream merge filters on it.
  Note the earliest-wins tie-break is right for a half-trained re-run and **wrong** for a
  superseded condition, which is older: that is why `es_es` had to leave serial 0 rather than be
  filtered out.
- **Aggregation.** `t6_loss_by_axis.csv` carries an `aggregation` column and reports **medians of
  per-language medians**, so a tier with seven languages cannot outvote one with two.
- **`audio_hours`** is cumulative audio **processed**, counting repeats. It is not unique corpus
  hours. Unique hours appear as `implied_stream_hours` in t4, always with an `estimate_kind` of
  `estimate` or `lower_bound`.
- **`stream_post_filter`** is the volume axis: example counts minus each config's at-cap
  fraction, because the strict `< 30 s` cap removes clips before the model sees them. For
  `ta_in` that is 32,107 → 8,846.
- **`in_domain_role`.** All 33 evaluated WorldSpeech variants normalise to the same study cell,
  so exactly one per cell is its in-domain point (`primary`) and the rest are
  `accent_transfer`. Hausa's primary is `ha_td`, not `ha_ng`, because `ha_ng test` was its
  training-time selection split.
- **t4 is not a data-integrity check.** A ratio away from 1.0 is a bookkeeping observation about
  the run. Integrity is checked only by `verify_dataset_durations.py`.
- **Absolute vs relative effect.** Both are reported. Volume and baseline CER are collinear
  (Spearman rho **-0.818**, p 0.0021), so the two candidate explanations are entangled: the
  partial correlation controlling for baseline CER is r **-0.129**, p **0.71**, inconclusive at
  8 residual degrees of freedom rather than null.
- **Why `ta_in` is not excluded.** The region-match contrast is within-language, so the cap loss
  cannot bias it. Excluding the language would remove the only low-resource cell.

## Check by eye after regenerating

- **Colours, not just positions and labels.** A `hue_order` containing a duplicate silently
  mis-assigns one category's colour while positions and legend text stay pixel-identical.
  `plot_curve.get_hue_order` de-duplicates, restricts to categories present, and **sorts** what
  it does not find in `METHODS_DIC` — it once iterated a `set`, and CPython randomises string
  hashing per process, so colours changed between renders while points stayed identical.
- **Facet count and captions against declared filters.** `--keep_states finished` drops
  unfinished runs, so a panel appears or disappears as runs land. A caption describing data the
  declared filter removed is a defect that shipped in a sibling repo — update the titles in
  `plotter.sh` when coverage changes.
- **symlog on the volume figures.** One language sits at -14.7 CER while the rest are within
  ±1.2. A linear y-axis compresses the rest into an unreadable band and a log axis cannot render
  a signed effect; `--symlog_y 1.0` is linear within ±1 and logarithmic outside.
- **Log axes.** Seaborn's line artists are not registered in the axes' data limits, so switching
  to log after drawing can leave a lower bound of 0 that matplotlib collapses, rendering an empty
  panel with no error. `plot_curve.py` sets limits from the positive data.

## Known-unverified inputs

- Whether the volume decay is driven by data volume or task difficulty — collinear, and only a
  within-language volume manipulation can separate them.
- The expected stream size for `crs_sc`, which trains on the `ERISLab/WorldSpeech` mirror; that
  split is not in the example-count snapshot.
- The matched-variant premise (identical base, tokenizer and parameter count across the four
  regional variants), still unconfirmed against the Tiny Aya report.

## Settled — do not re-report

- **Interleaving does not oversample the smaller config.** `all_exhausted_without_replacement`
  makes a combined stream exactly the sum of its parts. Proved offline in
  `verify_interleave_semantics.py` and confirmed on the real `ta_in` + `ta_lk` splits.
- **The Tamil configs contain no corrupt or mislabelled clips.** The duration-consistency filter
  removes zero samples. The Tamil loss was the duration cap, not data quality.
- **The `crs_sc` `_clean` splits are all genuinely cleaned** — train, val and test. The committed
  upstream example only demonstrates it on test, which is why it can look otherwise.
