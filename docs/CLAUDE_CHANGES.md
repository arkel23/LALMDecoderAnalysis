# Change log

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
  is a real measurement (`qasr/data/data_utils.py:71-82`), but `train/epoch` is an estimate
  under `streaming=True` and reconciles with the WorldSpeech report for only 3 of 9 languages
  (`ta_in` is off by ~7×). No table reports unique hours; everything says audio *processed*.
- **The endpoint comparison is underpowered.** Median within-run late-training CER sd is 1.03
  against a mean region effect of 0.48; minimum detectable effect is 6.97–9.27 CER.
- **Two confounds encoded in `TRAIN_EVAL_MATCH`**: `fr_fr`/`es_419` are cross-dialect
  (trained `fr_ca`/`es_es`, evaluated `fr_fr`/`es_419`), and `ta_in`/`ha_ng`/`sw_ke`
  interleave two WorldSpeech configs at 50/50 regardless of corpus size
  (`qasr/data/data_utils.py:239-251`).
- **`crs_sc`'s `val_clean` split is not duration-filtered** despite the name; only
  `test_clean` is. Recorded on `t3_crs_ood.csv`.

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
