#!/bin/bash

# SUPERSEDED for the main study -- read this before running it.
#
# This sweeps the UNTRAINED model configs: a stock Whisper encoder plus a stock Tiny Aya decoder
# with a randomly-initialised connector. That composition has never been trained to map audio
# into the decoder's space, so its transcripts are not a meaningful baseline; it measures the
# initialisation, not a model anyone would deploy.
#
# The baselines that matter are serial 10 -- whisper-medium, Voxtral-Mini and Qwen2-Audio
# evaluated directly -- and the trained checkpoints are serial 11
# (eval_lalm_decoder_txf.sh). Those two share FLEURS test configs, so 11 minus 10 is the
# like-for-like contrast.
#
# Kept rather than deleted because the dataset pairing below is still the reference for which
# eval set goes with which language, and because an untrained-connector row is occasionally
# worth having as a floor. Its serial (420) is outside the study's 10/11 range on purpose, so
# it cannot be confused with either.
#
# Original description follows.
#
# Evaluation sweep for the LALM decoder-SFT study: the five Tiny Aya decoder variants plus a
# non-Aya control, over the 12 study languages, on BOTH evaluation domains.
#
# WHERE THIS LIVES. Written in QuantizedASR's scripts/ style but kept in LALMDecoderAnalysis
# under for_quantizedasr/, because QuantizedASR is not modified from here. Copy
# for_quantizedasr/scripts/ and for_quantizedasr/tools/ into the QuantizedASR checkout, run
# the generator once, then run this from that repo's root.
#
# WHY BOTH DOMAINS. Training is always WorldSpeech (parliamentary / broadcast / audiobook),
# but 10 of the 12 languages are currently evaluated only on google/fleurs (read speech). So
# almost every number in the study is a domain-transfer number, and the design cannot separate
# "this decoder specialises better" from "this decoder transfers across domain better". This
# sweep evaluates every model on both:
#
#   FLEURS test        -- the held-out domain. NOTE: the training runs used FLEURS
#                         *validation*; `test` is used here because best-CER was selected over
#                         the validation curve, so validation is no longer untouched. Also
#                         matches the eval_short_ml*.sh convention of never using dev.
#   WorldSpeech test   -- the in-domain point, from create_yamls_worldspeech_lalm.py.
#
# Serial 420 is free: the short-ml family uses 300/301, 320/321, 340/341, 360/361, 380/381,
# and 400/500 are the long-form scripts. See the serial map in that repo's scripts/.
#
# Every FLEURS config below already exists (create_yamls_fleurs_full.py generates all 102
# languages x dev/test). Only the WorldSpeech ones are new.

DATASET_CONFIGS=(
    # --- held-out domain: FLEURS test, one per study language -----------------------
    "short_ml/fleurs_en_us_test.yaml"
    "short_ml/fleurs_fr_fr_test.yaml"
    "short_ml/fleurs_es_419_test.yaml"
    "short_ml/fleurs_hi_in_test.yaml"
    "short_ml/fleurs_id_id_test.yaml"
    "short_ml/fleurs_mr_in_test.yaml"
    "short_ml/fleurs_sw_ke_test.yaml"
    "short_ml/fleurs_ta_in_test.yaml"
    "short_ml/fleurs_ur_pk_test.yaml"
    "short_ml/fleurs_am_et_test.yaml"
    "short_ml/fleurs_ha_ng_test.yaml"

    # --- in-domain: WorldSpeech test, matching the training variety -----------------
    "short_ml/worldspeech_en_us_test.yaml"
    "short_ml/worldspeech_fr_ca_test.yaml"
    "short_ml/worldspeech_es_mx_test.yaml"
    "short_ml/worldspeech_hi_in_test.yaml"
    "short_ml/worldspeech_id_id_test.yaml"
    "short_ml/worldspeech_mr_in_test.yaml"
    "short_ml/worldspeech_sw_ke_test.yaml"
    "short_ml/worldspeech_ta_in_test.yaml"
    "short_ml/worldspeech_ur_pk_test.yaml"
    "short_ml/worldspeech_am_et_test.yaml"
    "short_ml/worldspeech_crs_sc_test.yaml"
)

# Seychellois Creole has no google/fleurs equivalent -- it is not a FLEURS language at all,
# which is precisely what makes it the study's out-of-distribution probe (unseen by both the
# Whisper encoder and every Tiny Aya decoder). It appears above on the WorldSpeech side only.
    # "short_ml/fleurs_crs_sc_test.yaml"

# ha_ng has no WorldSpeech entry above because the training runs ALREADY evaluate it on
# disco-eth/WorldSpeech test -- adding one here would duplicate an existing cell.
    # "short_ml/worldspeech_ha_ng_test.yaml"

MODEL_CONFIGS=(
    "cq2a_whisper_medium_tiny_aya_base.yaml"
    "cq2a_whisper_medium_tiny_aya_global.yaml"
    "cq2a_whisper_medium_tiny_aya_earth.yaml"
    "cq2a_whisper_medium_tiny_aya_fire.yaml"
    "cq2a_whisper_medium_tiny_aya_water.yaml"

    # Non-Aya control: separates "this is a Tiny Aya property" from "this is a
    # connector-recipe property". Currently run for crs_sc and ta_in only.
    "cq2a_whisper_medium_qwen_3_4b.yaml"
)

# Optional --models override: pass a single space-separated, quoted string of model config
# filenames (e.g. --models "cq2a_whisper_medium_tiny_aya_fire.yaml") to run only those.
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

base_cmd="python -m tools.evaluate --serial 420 --batch_size 128"

# Iterate through all combinations
for model_cfg in "${MODEL_CONFIGS[@]}"; do
    for dataset_cfg in "${DATASET_CONFIGS[@]}"; do
        # Execute the command
        cmd="$base_cmd --config configs/models/$model_cfg configs/datasets/$dataset_cfg"
        echo "$cmd"
        $cmd
    done
done
