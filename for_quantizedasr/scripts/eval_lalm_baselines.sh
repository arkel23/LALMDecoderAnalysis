#!/bin/bash

# Serial 10: off-the-shelf models over the study's languages, on both evaluation domains.
# No connector is trained, so this is a plain model x dataset cross product. Pairs with
# serial 11 (eval_lalm_decoder_txf.sh), which uses the same configs.

DATASET_CONFIGS=(
    # FLEURS: one variant per language -- verified, no prefix in the FLEURS list has two.
    # WorldSpeech: every variant of each study language, not only the trained one.
    "short_ml/fleurs_am_et_test.yaml"
    "short_ml/fleurs_en_us_test.yaml"
    "short_ml/fleurs_es_419_test.yaml"
    "short_ml/fleurs_fr_fr_test.yaml"
    "short_ml/fleurs_ha_ng_test.yaml"
    "short_ml/fleurs_hi_in_test.yaml"
    "short_ml/fleurs_id_id_test.yaml"
    "short_ml/fleurs_mr_in_test.yaml"
    "short_ml/fleurs_sw_ke_test.yaml"
    "short_ml/fleurs_ta_in_test.yaml"
    "short_ml/fleurs_ur_pk_test.yaml"
    "short_ml/worldspeech_am_et_test.yaml"
    "short_ml/worldspeech_crs_sc_test.yaml"
    "short_ml/worldspeech_en_au_test.yaml"
    "short_ml/worldspeech_en_jm_test.yaml"
    "short_ml/worldspeech_en_ke_test.yaml"
    "short_ml/worldspeech_en_nz_test.yaml"
    "short_ml/worldspeech_en_pk_test.yaml"
    "short_ml/worldspeech_en_sl_test.yaml"
    "short_ml/worldspeech_en_us_test.yaml"
    "short_ml/worldspeech_en_zm_test.yaml"
    "short_ml/worldspeech_es_ar_test.yaml"
    "short_ml/worldspeech_es_cl_test.yaml"
    "short_ml/worldspeech_es_co_test.yaml"
    "short_ml/worldspeech_es_es_test.yaml"
    "short_ml/worldspeech_es_mx_test.yaml"
    "short_ml/worldspeech_es_pe_test.yaml"
    "short_ml/worldspeech_es_pr_test.yaml"
    "short_ml/worldspeech_es_py_test.yaml"
    "short_ml/worldspeech_es_uy_test.yaml"
    "short_ml/worldspeech_fr_ca_test.yaml"
    "short_ml/worldspeech_fr_cd_test.yaml"
    "short_ml/worldspeech_fr_ci_test.yaml"
    "short_ml/worldspeech_ha_ng_test.yaml"
    "short_ml/worldspeech_ha_td_test.yaml"
    "short_ml/worldspeech_hi_in_test.yaml"
    "short_ml/worldspeech_id_id_test.yaml"
    "short_ml/worldspeech_mr_in_test.yaml"
    "short_ml/worldspeech_sw_ke_test.yaml"
    "short_ml/worldspeech_sw_tz_test.yaml"
    "short_ml/worldspeech_ta_in_test.yaml"
    "short_ml/worldspeech_ta_lk_test.yaml"
    "short_ml/worldspeech_ur_in_test.yaml"
    "short_ml/worldspeech_ur_pk_test.yaml"
)

MODEL_CONFIGS=(
    "whisper_tiny.yaml"
    "whisper_small.yaml"
    "whisper_medium.yaml"
    "whisper_large_v3_turbo.yaml"
    "voxtral_mini_3b.yaml"
    "qwen_2_audio_7b_instruct.yaml"
)

models=''

VALID_ARGS=$(getopt -o '' --long models: -- "$@")
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
    --) shift;
        break
        ;;
  esac
done

if [ -n "$models" ]; then
    read -ra MODEL_CONFIGS <<< "$models"
fi

base_cmd="python -m tools.evaluate --serial 10 --batch_size 128"

for model_cfg in "${MODEL_CONFIGS[@]}"; do
    for dataset_cfg in "${DATASET_CONFIGS[@]}"; do
        cmd="$base_cmd --config configs/models/$model_cfg configs/datasets/$dataset_cfg"
        echo "$cmd"
        $cmd
    done
done
