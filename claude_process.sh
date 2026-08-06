# One-off / ad-hoc commands run against this repo, with a dated comment each.
#
# This file is a running LOG, not a queue: once a line's purpose is served it is commented
# out rather than deleted, so `bash claude_process.sh` never re-runs spent work. Three live
# "DONE" lines would have relaunched a multi-hour recompute in a sibling repo.
#
# Permanent pipeline steps belong in plotter.sh, not here.

set -u

# --- 2026-07-30: negative-test the verification guards. RUN, all three went red. ----
# A checker that has never failed is not a checker.
#
# Lesson worth keeping: the first attempt perturbed row 0 of t1 and the checker stayed
# GREEN -- correctly, because t1 is sorted by (dataset, model_short) so row 0 is
# crs_sc/base, and every checked aggregate is computed over CORE_VARIANTS, which excludes
# base. A negative test has to perturb a value some check actually depends on, or it tests
# nothing. Retargeted below.
#
# Result: A and B fail on a drifted value, C breaks two ordering claims, and all three
# restore green. Spent -- leave commented; re-run by hand after changing either guard.
#
# bash -c '
#   set -e
#   T1=results_all/acc/t1_sample_efficiency.csv
#   T2=results_all/acc/t2_region_match_stats.csv
#   cp "$T1" /tmp/t1_backup.csv; cp "$T2" /tmp/t2_backup.csv
#
#   # A: move the checked median by shifting late_sd on the core-variant rows only.
#   python -c "
# import pandas as pd
# f=\"$T1\"; d=pd.read_csv(f)
# m=d.model_short.isin([\"earth\",\"fire\",\"global\",\"water\"])
# d.loc[m,\"late_sd\"]=d.loc[m,\"late_sd\"]+5.0
# d.to_csv(f,index=False,float_format=\"%.6f\")"
#   python verify_paper_numbers.py && echo "BAD: passed on perturbed data" || echo "ok: red"
#   cp /tmp/t1_backup.csv "$T1"
#
#   # B: drift a Wilcoxon p in the stats CSV.
#   python -c "
# import pandas as pd
# f=\"$T2\"; d=pd.read_csv(f); d.loc[0,\"wilcoxon_p\"]=0.001
# d.to_csv(f,index=False,float_format=\"%.6f\")"
#   python verify_paper_numbers.py && echo "BAD: passed" || echo "ok: red"
#   cp /tmp/t2_backup.csv "$T2"
#
#   # C: break an ordering claim by making global the fastest variant.
#   python -c "
# import pandas as pd
# f=\"$T1\"; d=pd.read_csv(f)
# d.loc[d.model_short==\"global\",\"audio_h_to_1.5x_best\"]=1.0
# d.to_csv(f,index=False,float_format=\"%.6f\")"
#   python verify_paper_numbers.py && echo "BAD: passed" || echo "ok: orderings broken"
#   cp /tmp/t1_backup.csv "$T1"
#
#   python verify_paper_numbers.py | tail -3
# '

# --- 2026-07-30: confirm the region map is load-bearing. ----------------------------
# Re-tiering a single language must break the ordering claims, otherwise the checker is not
# actually testing the region logic. Spent -- leave commented.
#
# bash -c '
#   cp utils.py /tmp/utils_backup.py
#   sed -i "s/^    .ta_in.: .fire.,/    \"ta_in\": \"water\",/" utils.py
#   python analyze_region_match.py > /dev/null 2>&1 || true
#   echo "--- expecting BROKEN orderings with ta_in mis-tiered ---"
#   python verify_paper_numbers.py 2>&1 | grep -E "broken|BROKEN" | head -5
#   cp /tmp/utils_backup.py utils.py
#   python analyze_region_match.py > /dev/null && python verify_paper_numbers.py 2>&1 | grep "broken"
# '

# --- 2026-07-30: verify the Tamil training configs directly. RUN, both checks passed. ------
# Settles a suspicion an earlier pass wrote down without testing: that uniform interleave
# probabilities oversampled the smaller config, and that a duration-column inconsistency was
# silently dropping Tamil samples. Neither is true.
#   len(interleaved) == len(ta_in) + len(ta_lk)   -- interleaving loses nothing
#   duration-consistency filter removed 0 samples -- no corrupt or mislabelled clips
# Downloads ~7 GB of audio, so it is opt-in and NOT in plotter.sh (which runs the instant
# metadata-only mode instead). MUST run in the `asr` env: `pytorch` has no audio backend.
# Spent -- leave commented.
#
# ~/miniconda3/envs/asr/bin/python -u verify_dataset_durations.py \
#   --dataset_path disco-eth/WorldSpeech --dataset_configs ta_in ta_lk --split train \
#   --load --num_proc 20
#
# Independent confirmation on the test splits, run here 2026-07-30 (~0.4 GB, ~4 min):
# ALL CHECKS PASSED -- 1690 == 466 + 1224, 0 undecodable, 0 removed, 12.00 audio hours.
# ~/miniconda3/envs/asr/bin/python -u verify_dataset_durations.py \
#   --dataset_path disco-eth/WorldSpeech --dataset_configs ta_in ta_lk --split test \
#   --load --num_proc 8

# --- 2026-07-30: refresh the example-count snapshot in utils.py. --------------------------
# Metadata only, no audio. Re-run if a training config is added or WorldSpeech is revised, and
# paste the counts into utils.WORLDSPEECH_TRAIN_EXAMPLES.
#
# for cfg in en_us fr_ca es_es es_mx hi_in sw_ke sw_tz ha_ng ha_td ta_in ta_lk mr_in id_id; do
#   python -u verify_dataset_durations.py --dataset_path disco-eth/WorldSpeech \
#     --dataset_configs "$cfg" --split train
# done

# --- 2026-07-30: screen every WorldSpeech training config for clips at the duration cap. ----
# This is what found the ta_lk loss: 100/100 sampled clips at exactly 30.00 s, against a strict
# `< 30` filter, so the whole config is deleted. Metadata only, no audio downloaded. Results are
# frozen into utils.CONFIG_DURATION_AT_CAP; re-run if a config is added or WorldSpeech is revised.
# Spent -- leave commented. Note the endpoint 500s for uncached configs, which is why the
# snapshot, not the endpoint, is authoritative in the checker.
#
# for cfg in en_us fr_ca es_es es_mx hi_in sw_ke sw_tz ha_ng ha_td ta_in ta_lk mr_in id_id; do
#   python -u verify_dataset_durations.py --dataset_path disco-eth/WorldSpeech \
#     --dataset_configs "$cfg" --split train --known_at_cap
# done

# --- 2026-07-30: negative-test the at-cap screen. RUN, both directions correct. -------------
# ta_lk must FAIL when not acknowledged, and hi_in must PASS. Spent -- leave commented.
#
# python -u verify_dataset_durations.py --dataset_path disco-eth/WorldSpeech \
#   --dataset_configs ta_lk --split train --known_at_cap   # expect exit 1
# python -u verify_dataset_durations.py --dataset_path disco-eth/WorldSpeech \
#   --dataset_configs hi_in --split train                  # expect exit 0

# --- 2026-07-30: prove plot_curve's hue order is deterministic. RUN, byte-identical. --------
# It previously iterated a set, so region colours changed between renders while positions stayed
# identical. Spent -- leave commented; re-run after touching get_hue_order.
#
# for i in 1 2; do
#   python -u plot_curve.py --input_file results_all/acc/t5_volume_interaction.csv \
#     --kind scatter --x_var_name stream_post_filter --y_var_name delta_vs_mismatched \
#     --hue_var_name region --annotate_var_name dataset --hline 0 --log_scale_x --symlog_y 1.0 \
#     --legend_outside --output_file s0/s0_volume_interaction --results_dir results_all/plots
#   md5sum results_all/plots/s0/s0_volume_interaction.png
# done

# --- 2026-08-01: move the superseded en_us/water run off serial 0. RUN, applied. -----------
# The cell was run twice (n4cot5v7 2026-07-27 best 17.06; pwnz2zno 2026-07-31 best 12.05),
# which broke the one-row-per-(model, language) contract every analysis table asserts. The
# ORIGINAL moves to serial 1 so serial 0 stays the canonical grid and the superseded run stays
# retrievable. Dry run first (the default), then --execute. MUTATES a live W&B project.
# Spent -- leave commented.
#
# python -u rename_wandb_serial.py --run_ids n4cot5v7 --from_serial 0 --to_serial 1
# python -u rename_wandb_serial.py --run_ids n4cot5v7 --from_serial 0 --to_serial 1 --execute

# --- 2026-08-01: screen the two new languages for the 30 s cap. RUN, both clean. -----------
# am_et 0/100 at the cap (mean 15.36 s), ur_pk 0/100 (mean 8.36 s). Spent -- leave commented.
#
# for cfg in am_et ur_pk ur_in; do
#   python -u verify_dataset_durations.py --dataset_path disco-eth/WorldSpeech \
#     --dataset_configs "$cfg" --split train
# done

# --- 2026-08-01: generate the WorldSpeech in-domain eval configs. NOT YET RUN in ------------
# QuantizedASR. These belong to that repo; developed here under for_quantizedasr/ because
# QuantizedASR is not modified from this session. Copy for_quantizedasr/tools/ and
# for_quantizedasr/scripts/ into that checkout, then run from its root:
#
#   python tools/preprocess/create_yamls_worldspeech_lalm.py   # writes 11 configs
#   bash scripts/eval_lalm_decoder.sh                          # serial 420, both domains

# --- 2026-08-02: generate model configs for the trained checkpoints. NOT YET RUN in ---------
# QuantizedASR. Crawls the ERISLab HF org (86 checkpoints: 5 variants x 11 training languages
# x best-step + step-1000) and writes configs/models/*_txf_*.yaml. Skips es_es (superseded by
# es_mx; no step-1000 exists). Reports that am_et and crs_sc have no uploaded checkpoints yet.
# Copy for_quantizedasr/ into the QuantizedASR checkout, then from its root:
#
#   python tools/preprocess/create_yamls_models_lalm_txf.py   # 82 model configs
#   python tools/preprocess/create_yamls_worldspeech_lalm.py  # 11 dataset configs
#   bash scripts/eval_lalm_decoder_txf.sh                     # serial 11, 156 evals

# --- 2026-08-03: regenerate the eval configs, the manifest AND the pairings. SPENT ----------
# (run locally, inside this repo, only to refresh the tracked manifest + pairings file).
# The generator now also emits worldspeech_lalm_pairings.sh, which eval_lalm_decoder_txf.sh
# sources instead of carrying an inline PAIRINGS array -- so a language is added in one place
# and a cross-language pairing is impossible by construction.
#
# It writes its YAMLs to a cwd-relative path (correct in QuantizedASR, where it runs from the
# repo root), so running it here also drops a configs/ tree next to the generator. That tree is
# gitignored; the tracked, reviewable artifacts are the generator, the manifest and the pairings.
#
# (cd for_quantizedasr/tools/preprocess && python create_yamls_worldspeech_lalm.py)

# --- 2026-08-03: SUPERSEDES the 2026-08-02 block above -- 12 pairings, not 11, and the -------
# eval count depends on --eval_set. NOT YET RUN in QuantizedASR. From its root:
#
#   python tools/preprocess/create_yamls_models_lalm_txf.py    # 48 model configs (best step,
#                                                              # 4 grid variants x 12 languages)
#   python tools/preprocess/create_yamls_worldspeech_lalm.py   # 120 dataset configs
#   bash scripts/eval_lalm_decoder_txf.sh --eval_set primary   # serial 11, 104 evals (26/variant)
#   bash scripts/eval_lalm_decoder_txf.sh --eval_set all        # serial 11, 176 evals (44/variant)
#   bash scripts/eval_lalm_decoder_txf.sh --models fire --eval_set all   # one variant, 44 evals
#   bash scripts/eval_lalm_baselines.sh                         # serial 10, 264 evals
#
# NOTE: the earlier version of the generator wrote 104 configs including step-1000 and the
# controls. If those are already in configs/models/, delete them -- the sweep skips `_1k` but
# the control configs would just sit there unused.

# --- 2026-08-03: PENDING, do NOT run until the crs_sc/am_et re-runs FINISH. -----------------
# The 5 crs_sc re-runs (seed 420) are currently in serial 0 alongside the finished originals
# (seed 42). analyze_sample_efficiency.py keeps the pipeline correct meanwhile by marking the
# finished original canonical, and prints a [DUPLICATE RUNS] warning on every run.
#
# Once they finish, migrate the ORIGINALS to serial 1 so replicate pairing is a clean join on
# (model_id, dataset). Dry-run first; the script defaults to dry-run.
#
#   python rename_wandb_serial.py --run_ids 12sof3iv g8f56u6p wvi4k6py ea7a4ud8 95kqse0h \
#       --from_serial 0 --to_serial 1              # dry run: prints what would change
#   python rename_wandb_serial.py --run_ids 12sof3iv g8f56u6p wvi4k6py ea7a4ud8 95kqse0h \
#       --from_serial 0 --to_serial 1 --execute    # then: rm data/raw_serials/history_serial_*.csv
#                                                  #       bash plotter.sh

# --- 2026-08-05: the am_et/crs_sc re-runs finished and their checkpoints are on HF, so the -----
# originals move to serial 1. 4 am_et (earth/fire/global/water) + 5 crs_sc (those plus base),
# all seed 42; the replacements are seed 420. SPENT.
#
# python rename_wandb_serial.py \
#     --run_ids wd3s5858 p7yi3bau u1t6bx57 8fupnxu6 12sof3iv g8f56u6p wvi4k6py ea7a4ud8 95kqse0h \
#     --from_serial 0 --to_serial 1 --execute

# --- 2026-08-05: split serial 0 into a clean 12x4 grid. SPENT. -------------------------------
# 0 -> 2  the control arms (base, qwen3-4b on crs_sc and ta_in)
# 1 -> 3  the superseded control (crs_sc/base, seed 42)
# 1 -> 4  the same-seed replicate (en_us/water; the re-run set no new seed)
# 0 -> 5  the es_es-trained runs, re-added from another project and superseded by es_mx
#
# python rename_wandb_serial.py --run_ids dv5swzra hxk4nq63 o1aorqtl w68nxmia \
#     --from_serial 0 --to_serial 2 --execute
# python rename_wandb_serial.py --run_ids 12sof3iv --from_serial 1 --to_serial 3 --execute
# python rename_wandb_serial.py --run_ids n4cot5v7 --from_serial 1 --to_serial 4 --execute
# python rename_wandb_serial.py --run_ids 7z4munf2 31wgxqpp e8h3ys8j xejcc9qt \
#     --from_serial 0 --to_serial 5 --execute
