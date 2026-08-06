#!/bin/bash

# Serial 11: the trained connector checkpoints, each evaluated only on the language it was
# trained on. Pairs with serial 10 (eval_lalm_baselines.sh) on the same configs.
#
# training stem | FLEURS test | in-training WorldSpeech | held-out WorldSpeech
#
# fr_ca -> fleurs_fr_fr and es_mx -> fleurs_es_419: trained variety != FLEURS variety.
# ta_lk is held-out; the strict 30 s cap dropped all of it. crs_sc has no FLEURS entry.
#
# ha_ng's checkpoint was SELECTED on worldspeech_ha_ng_test during training, so that
# number is not held out. Use ha_td as Hausa's in-domain point -- same training mix,
# never used for selection.
PAIRINGS=(
    "en_us|short_ml/fleurs_en_us_test.yaml|short_ml/worldspeech_en_us_test.yaml|short_ml/worldspeech_en_au_test.yaml short_ml/worldspeech_en_jm_test.yaml short_ml/worldspeech_en_ke_test.yaml short_ml/worldspeech_en_nz_test.yaml short_ml/worldspeech_en_pk_test.yaml short_ml/worldspeech_en_sl_test.yaml short_ml/worldspeech_en_zm_test.yaml"
    "es_mx|short_ml/fleurs_es_419_test.yaml|short_ml/worldspeech_es_mx_test.yaml|short_ml/worldspeech_es_ar_test.yaml short_ml/worldspeech_es_cl_test.yaml short_ml/worldspeech_es_co_test.yaml short_ml/worldspeech_es_es_test.yaml short_ml/worldspeech_es_pe_test.yaml short_ml/worldspeech_es_pr_test.yaml short_ml/worldspeech_es_py_test.yaml short_ml/worldspeech_es_uy_test.yaml"
    "fr_ca|short_ml/fleurs_fr_fr_test.yaml|short_ml/worldspeech_fr_ca_test.yaml|short_ml/worldspeech_fr_cd_test.yaml short_ml/worldspeech_fr_ci_test.yaml"
    "ha_ng|short_ml/fleurs_ha_ng_test.yaml|short_ml/worldspeech_ha_ng_test.yaml short_ml/worldspeech_ha_td_test.yaml|"
    "sw_ke|short_ml/fleurs_sw_ke_test.yaml|short_ml/worldspeech_sw_ke_test.yaml short_ml/worldspeech_sw_tz_test.yaml|"
    "ur_pk|short_ml/fleurs_ur_pk_test.yaml|short_ml/worldspeech_ur_pk_test.yaml short_ml/worldspeech_ur_in_test.yaml|"
    "ta_in|short_ml/fleurs_ta_in_test.yaml|short_ml/worldspeech_ta_in_test.yaml|short_ml/worldspeech_ta_lk_test.yaml"
    "hi_in|short_ml/fleurs_hi_in_test.yaml|short_ml/worldspeech_hi_in_test.yaml|"
    "id_id|short_ml/fleurs_id_id_test.yaml|short_ml/worldspeech_id_id_test.yaml|"
    "mr_in|short_ml/fleurs_mr_in_test.yaml|short_ml/worldspeech_mr_in_test.yaml|"
    "am_et|short_ml/fleurs_am_et_test.yaml|short_ml/worldspeech_am_et_test.yaml|"
    "crs_sc||short_ml/worldspeech_crs_sc_test.yaml|"
)

# Decoder slugs, matching create_yamls_models_lalm_txf.py's filenames. tiny_aya_base and
# qwen3_4b exist for ta_in and crs_sc only; the glob below skips the rest.
VARIANTS=(tiny_aya_earth tiny_aya_fire tiny_aya_global tiny_aya_water tiny_aya_base qwen3_4b)

# --eval_set primary  FLEURS + in-training WorldSpeech (default)
# --eval_set all      adds the held-out variants (the accent-transfer axis)
models=''
eval_set='primary'

VALID_ARGS=$(getopt -o '' --long models:,eval_set: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1;
fi

eval set -- "$VALID_ARGS"
while [ : ]; do
  case "$1" in
    --models)
        models=${2}
        shift 2
        ;;
    --eval_set)
        eval_set=${2}
        shift 2
        ;;
    --) shift;
        break
        ;;
  esac
done

if [ -n "$models" ]; then
    read -ra VARIANTS <<< "$models"
fi

base_cmd="python -m tools.evaluate --serial 11 --batch_size 128 \
    --wandb_entity LisTAya \
    --wandb_project LALMDecoder"

for pairing in "${PAIRINGS[@]}"; do
    IFS='|' read -r lang fleurs_cfg ws_in ws_held <<< "$pairing"

    eval_cfgs="$fleurs_cfg $ws_in"
    if [ "$eval_set" = "all" ]; then
        eval_cfgs="$eval_cfgs $ws_held"
    fi

    for variant in "${VARIANTS[@]}"; do
        # Both checkpoints per cell: the best step (which differs per cell, hence the glob)
        # and step 1000.
        for model_cfg_path in configs/models/cq2a_whisper_medium_${variant}_txf_ws_${lang}_*.yaml; do
            [ -e "$model_cfg_path" ] || continue

            for dataset_cfg in $eval_cfgs; do
                cmd="$base_cmd --config configs/models/$(basename "$model_cfg_path") configs/datasets/$dataset_cfg"
                echo "$cmd"
                $cmd
            done
        done
    done
done
