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
