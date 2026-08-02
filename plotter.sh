# LALMDecoderAnalysis figure/table generation. Invoke from the repo root: `bash plotter.sh`
#
# One command rebuilds everything from a bare checkout (code + docs, no data/*.csv): both
# wandb downloads, every analysis CSV, every figure, and the verification guards. There is
# no "run this by hand first" step, deliberately -- retrofitting that property was expensive
# in the sibling repos.
#
# Run with the `pytorch` conda env already active (matching the sibling repos, which also do
# not activate it themselves).

# -e matters: without it this script's exit status is only the LAST command's, so a failing
# verification guard in section 5 would be printed and then silently ignored. That happened
# once during development -- test_utils_port.py exited 1 and plotter.sh still reported success.
set -eu

SERIALS=(0)
EVAL_SERIALS=(10 11)  # 10 = off-the-shelf baselines, 11 = trained TinyAya LALMs
PROJECT='LisTAya/LALMDecoder'
HIST=data/raw_serials/history_serial_0.csv
PLOTS=results_all/plots/s0
ACC=results_all/acc

mkdir -p data/raw_serials "$ACC" "$PLOTS" logs

# --- 1. Downloads ------------------------------------------------------------------
# Guarded on file existence: both are network calls, and re-running plotter.sh on an
# already-populated repo must not re-download. Delete the file to force a refresh.
#
# One serial per invocation -- a combined `--serials` query builds a much slower $or filter.

# Summary: one row per run. This is the shape eval-only runs produce, so the wer / cer /
# rtfx / no_params columns stay in the request even though serial 0 is training runs and
# leaves them empty. The HF-style keys (eval/cer, train/loss, ...) are what serial 0 fills.
for serial in "${SERIALS[@]}"; do
  if [ ! -f "data/raw_serials/raw_serial_${serial}.csv" ]; then
    echo "Downloading summary for serial ${serial}..."
    python -u download_save_wandb_data.py \
      --project_name "$PROJECT" --serials "${serial}" \
      --results_dir data --output_file "raw_serials/raw_serial_${serial}.csv" \
      --config_cols serial dataset_path dataset split model_id force_asr_language \
                    'model/num_parameters' batch_size gradient_accumulation_steps \
                    max_steps lr \
      --summary_cols wer cer rtfx no_params num_samples audio_length_s_mean \
                     'eval/cer' 'eval/loss' 'train/loss' 'train/train_audio_seconds' \
                     'train/epoch' 'train/global_step' total_flos train_runtime
  else
    echo "Skipping summary download for serial ${serial} (exists)"
  fi
done

# History: one row per (run, step). ~68 s and ~7.5k rows for serial 0's 41 runs.
# --history is the flag that switches the time-series download on; without it the script
# is a no-op, so the intent is explicit at the call site.
for serial in "${SERIALS[@]}"; do
  if [ ! -f "data/raw_serials/history_serial_${serial}.csv" ]; then
    echo "Downloading history for serial ${serial}..."
    python -u download_wandb_history.py \
      --project_name "$PROJECT" --serials "${serial}" --history \
      --results_dir data --output_file "raw_serials/history_serial_${serial}.csv"
  else
    echo "Skipping history download for serial ${serial} (exists)"
  fi
done

# Serial 10: eval-only baselines (whisper-medium / Voxtral-Mini / Qwen2-Audio evaluated
# directly, no connector trained). These runs have _step=0, so there is no history worth
# pulling -- only the summary. This is exactly the run shape the wer / mer / wil / wip / rtfx /
# no_params column list was kept for; serial 0's training runs leave all of them empty.
#
# NOT guarded on file existence: serial 10 is being filled in incrementally, so a cached copy
# goes stale as runs land. It is one cheap API call.
for serial in "${EVAL_SERIALS[@]}"; do
  echo "Downloading eval-only summary for serial ${serial}..."
  python -u download_save_wandb_data.py \
    --project_name "$PROJECT" --serials "${serial}" \
    --results_dir data --output_file "raw_serials/raw_serial_${serial}.csv" \
    --config_cols serial dataset_path dataset split model_id force_asr_language task \
                  batch_size max_eval_samples norm_english long_form \
                  'model/num_parameters' \
    --summary_cols wer mer wil wip cer rtfx no_params n_params num_samples \
                   audio_length_s_mean audio_length_s_std audio_length_s_min \
                   audio_length_s_max max_memory bpw total_MB \
    || echo "  (serial ${serial} not available yet -- skipping)"
done

# --- 1b. Dataset example counts ----------------------------------------------------
# Authoritative num_examples per training config, straight from the dataset builder metadata
# (no audio downloaded). This is what the accounting compares against -- it replaced an
# hours snapshot recovered from a summarised reading of the WorldSpeech paper, which was the
# weakest input in the analysis and produced a false data-integrity finding.
#
# Pass --load to additionally download audio and run the duration-consistency and interleave
# assertions. That is opt-in because it is tens of GB; it has been run for the Tamil pair and
# both checks passed (interleave lossless, zero samples removed).
# Also screens for the strict-cap data-loss bug: it reports ta_lk's 100%-at-cap share as a
# KNOWN, documented issue (docs/UPSTREAM_FIXES.md) rather than failing, so the guard gates on
# NEW regressions. Snapshot-first, so it works offline and deterministically.
if [ ! -f "data/dataset_checks/disco-eth_WorldSpeech_ta_in-ta_lk_train.csv" ]; then
  python -u verify_dataset_durations.py --dataset_path disco-eth/WorldSpeech \
    --dataset_configs ta_in ta_lk --split train
else
  echo "Skipping dataset metadata check (exists)"
fi

# --- 1c. Tiny Aya post-training composition ----------------------------------------
# Parses the report's Appendix A tables into a CSV, turning "specialisation" from a
# categorical label into a continuous per-language exposure variable. Cached after the first
# fetch; delete data/tinyaya_report/ to re-download.
if [ ! -f "data/tinyaya_report/tinyaya_language_composition_wide.csv" ]; then
  python -u fetch_tinyaya_composition.py
else
  echo "Skipping Tiny Aya composition fetch (exists)"
fi

# --- 2. Sanity gate ----------------------------------------------------------------
# Cheap, and it runs before anything derives numbers: per-dataset run counts and state
# breakdowns, so a half-finished grid is visible rather than silently averaged.
python -u count_data.py \
  --input_file data/raw_serials/raw_serial_0.csv \
  --results_dir "$ACC" --output_file count_serial_0.csv

# --- 3. Analysis tables ------------------------------------------------------------
# Cheap derivations, deliberately NOT guarded: they must stay consistent with whatever is
# currently in data/, or a stale table outlives the correction that invalidated it.
#
# t1 keeps every run including the unfinished ones (state is a column); downstream scripts
# filter. Keeping raw completeness here means a run finishing does not require re-deriving
# from wandb.
python -u analyze_sample_efficiency.py \
  --input_file "$HIST" --output_file "$ACC/t1_sample_efficiency.csv"

# Data accounting before the contrasts, so t2 can carry each language's verdict. Derived
# purely from logged scalars plus the frozen WorldSpeech snapshot in utils.py -- this repo
# does analysis only and never loads a dataset.
python -u analyze_data_accounting.py \
  --input_file "$HIST" --output_file "$ACC/t4_data_accounting.csv" \
  --per_language_file "$ACC/t4_data_accounting_by_language.csv"

python -u analyze_region_match.py \
  --input_file "$ACC/t1_sample_efficiency.csv" \
  --output_file "$ACC/t2_region_match.csv" \
  --stats_file "$ACC/t2_region_match_stats.csv" \
  --metric best_cer

python -u analyze_ood_crs.py \
  --input_file "$ACC/t1_sample_efficiency.csv" \
  --output_file "$ACC/t3_crs_ood.csv"

# Baselines: off-the-shelf LALMs evaluated directly, answering the prior question serial 0
# cannot -- whether connector training is worth doing at all. The training-vs-baseline contrast
# waits for serial 11 (the trained checkpoints over the same FLEURS test configs).
python -u analyze_baselines.py \
  --input_file data/raw_serials/raw_serial_10.csv \
  --trained_file data/raw_serials/raw_serial_11.csv \
  --output_file "$ACC/t7_baselines.csv" \
  --contrast_file "$ACC/t7_training_vs_baseline.csv" || true

# The headline analysis: does the region-match benefit depend on training-data volume? Consumes
# t2 (per-language contrasts) and t4 (reconstructed stream sizes), so it runs after both.
# Loss diagnostics: separates overfitting (eval loss rising after its own minimum) from
# domain/accent shift (a large train-eval gap with no rise). Both curves are already logged.
python -u analyze_loss_metrics.py \
  --input_file "$HIST" --output_file "$ACC/t6_loss_metrics.csv" \
  --summary_file "$ACC/t6_loss_by_axis.csv"

python -u analyze_volume_interaction.py \
  --region_file "$ACC/t2_region_match.csv" \
  --accounting_file "$ACC/t4_data_accounting_by_language.csv" \
  --output_file "$ACC/t5_volume_interaction.csv" \
  --stats_file "$ACC/t5_volume_stats.csv"

# How large is the "specialisation" treatment, in percentage points of post-training data?
# Without this the region-match null cannot be distinguished from a null caused by a treatment
# too small to measure.
python -u analyze_exposure.py \
  --volume_file "$ACC/t5_volume_interaction.csv" \
  --output_file "$ACC/t8_exposure.csv" --stats_file "$ACC/t8_exposure_stats.csv"


# --- 4. Figures --------------------------------------------------------------------
# mkdir -p on the plot subdir happened above: plot.py only creates --results_dir, so a
# subdirectory inside --output_file raises FileNotFoundError on a bare checkout.
#
# Only finished runs are plotted: a still-training run's curve is truncated and would read
# as a variant that stopped improving early.

# The headline curve: eval CER against audio processed, per decoder variant, per language.
python -u plot_curve.py --input_file "$HIST" \
  --x_var_name audio_hours --y_var_name 'eval/cer' \
  --hue_var_name model_id --facet_var_name dataset --keep_states finished \
  --col_wrap 5 --log_scale_x --log_scale_y --font_scale 0.85 \
  --title 'Eval CER vs audio processed, by decoder variant' \
  --output_file s0/s0_curve_cer_vs_audio_hours --results_dir results_all/plots

# Eval loss on the same axis: separates "the connector is still learning" from "CER is noisy".
python -u plot_curve.py --input_file "$HIST" \
  --x_var_name audio_hours --y_var_name 'eval/loss' \
  --hue_var_name model_id --facet_var_name dataset --keep_states finished \
  --col_wrap 5 --log_scale_x --font_scale 0.85 \
  --title 'Eval loss vs audio processed, by decoder variant' \
  --output_file s0/s0_curve_evalloss_vs_audio_hours --results_dir results_all/plots

# Training loss, same treatment.
python -u plot_curve.py --input_file "$HIST" \
  --x_var_name audio_hours --y_var_name 'train/loss' \
  --hue_var_name model_id --facet_var_name dataset --keep_states finished \
  --col_wrap 5 --log_scale_x --font_scale 0.85 \
  --title 'Training loss vs audio processed, by decoder variant' \
  --output_file s0/s0_curve_trainloss_vs_audio_hours --results_dir results_all/plots

# The crs_sc OOD cell on its own, un-faceted. It is the only language run with six models,
# but the non-Aya Qwen3-4B control is still training, and --keep_states finished drops it --
# so the title says five, not six. A caption that describes data the declared filter removed
# is exactly the mismatch that shipped in a sibling repo; update it when Qwen3-4B finishes.
python -u plot_curve.py --input_file "$HIST" \
  --x_var_name audio_hours --y_var_name 'eval/cer' --hue_var_name model_id \
  --keep_languages crs_sc --keep_states finished \
  --log_scale_x --log_scale_y --fig_size 8 5 \
  --title 'Seychellois Creole, unseen by encoder AND decoder\n(5 finished TinyAya variants; Qwen3-4B control still training)' \
  --output_file s0/s0_curve_crs_ood --results_dir results_all/plots

# The headline figure. symlog on y because one language sits at -14.7 CER while the rest are
# within +/-1.2: a linear axis compresses six of seven points into an unreadable band, and a log
# axis cannot render a signed effect at all.
python -u plot_curve.py --input_file "$ACC/t5_volume_interaction.csv" \
  --kind scatter --x_var_name stream_post_filter --y_var_name delta_vs_mismatched \
  --hue_var_name region --annotate_var_name dataset --hline 0 \
  --log_scale_x --symlog_y 1.0 --legend_outside --fig_size 7.5 4.6 --font_scale 0.9 \
  --title 'Region-matched decoder benefit decays with training-data volume' \
  --output_file s0/s0_volume_interaction --results_dir results_all/plots

# The same effect relative to baseline CER. This is the confound made visible rather than
# argued: data volume and baseline error rate are collinear across these languages, so the
# relative view is what a reader needs to judge the claim.
python -u plot_curve.py --input_file "$ACC/t5_volume_interaction.csv" \
  --kind scatter --x_var_name stream_post_filter \
  --y_var_name relative_delta_vs_mismatched_pct \
  --hue_var_name region --annotate_var_name dataset --hline 0 \
  --log_scale_x --symlog_y 1.0 --legend_outside --fig_size 7.5 4.6 --font_scale 0.9 \
  --title 'Same effect as a share of baseline CER (the difficulty confound)' \
  --output_file s0/s0_volume_interaction_relative --results_dir results_all/plots

# --- 5. Verification guards --------------------------------------------------------
# Always last, and they fail loudly rather than letting an inconsistency reach a document.
#
# Unit tests first: they need no data for sections 1-5, run in ~1 s, and check the LOGIC.
# verify_paper_numbers.py cannot catch a CSV that is CONSISTENTLY wrong -- a mis-regioned
# language would regenerate every table wrongly and still pass every numeric check.
python -u test_utils_port.py
# Offline and instant: proves the multi-config interleaving reads every example exactly once,
# so "uniform probabilities oversample the smaller config" stays refuted rather than becoming
# folklore again.
python -u verify_interleave_semantics.py
python -u verify_paper_numbers.py
