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
