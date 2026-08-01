"""
Generates `disco-eth/WorldSpeech` *test*-split eval configs for the LALM decoder study's
12 languages.

WHERE THIS LIVES AND WHY. This file is written in QuantizedASR's `tools/preprocess/` style
but kept in LALMDecoderAnalysis under `for_quantizedasr/`, because QuantizedASR is not to be
modified from here. Copy `for_quantizedasr/tools/preprocess/` and `for_quantizedasr/scripts/`
into the QuantizedASR checkout to use them; the relative `output_dir` below is written so it
resolves correctly when run from the QuantizedASR repo root, exactly like the generators it
mirrors.

WHY THESE CONFIGS ARE NEEDED. Every model in the study trains on WorldSpeech, but 10 of the
12 languages are evaluated on `google/fleurs` — so almost every reported number is a
domain-transfer number (parliamentary/broadcast -> read speech) rather than an in-domain one.
Only `ha_ng` (disco-eth/WorldSpeech test) and `crs_sc` (ERISLab/WorldSpeech val_clean) are
in-domain today. Adding the matched WorldSpeech `test` split for every language gives each
one an in-domain point beside its FLEURS point, which is what separates "this decoder
specialises better" from "this decoder transfers across domain better".

WorldSpeech defines a 95/5 train/test split per country-language pair, so a `test` split
exists for every config trained on.

THE la_va/si_lk/tl_ph DECODE NOTE IS STALE. The three existing WorldSpeech configs in
QuantizedASR carry a comment saying every example hits a libsndfile/Opus decode error. That
was an environment limitation, not a corpus problem, and it no longer applies in an env built
from that repo's own requirements.txt: with `torchcodec==0.9.1` installed, `datasets` 4.5.0
decodes Audio through torchcodec/FFmpeg and never touches libsndfile. Retested 2026-07-30 --
la_va 29.54 s, si_lk 29.34 s, tl_ph 0.80 s all decode correctly at 24 kHz. See
docs/UPSTREAM_FIXES.md.

CAUTION -- the 30 s cap. `ta_lk` is pre-segmented into fixed 30.00 s windows, and the training
filter keeps a clip only when `length < max_input_length`, so a `max_input_length: 30` run
discards that config entirely. That is a *training*-side filter and does not affect these eval
configs, but the same trap applies to any future training config, so screen a new config with
LALMDecoderAnalysis's `verify_dataset_durations.py` before trusting it.

Usage (from the QuantizedASR repo root, after copying):
    python tools/preprocess/create_yamls_worldspeech_lalm.py
"""
import os

import yaml

# (WorldSpeech config, force_asr_language, note)
#
# One entry per language cell in the study. Where a cell trains on two configs, the eval
# entry is the config the study actually evaluates on, so the in-domain eval matches the
# reported cell rather than the union.
WORLDSPEECH_ENTRIES = [
    ('en_us',  'en', 'English - study cell en_us'),
    ('fr_ca',  'fr', 'French - the study TRAINS on fr_ca and evaluates FLEURS fr_fr, so the '
                     'in-domain point must be fr_ca to match the training variety'),
    ('es_mx',  'es', 'Spanish - study trains es_mx (the es_es runs were deleted 2026-08-01)'),
    ('hi_in',  'hi', 'Hindi - study cell hi_in'),
    ('id_id',  'id', 'Indonesian - study cell id_id'),
    ('mr_in',  'mr', 'Marathi - study cell mr_in'),
    ('sw_ke',  'sw', 'Swahili - study cell sw_ke (trained sw_ke + sw_tz)'),
    ('ta_in',  'ta', 'Tamil - study cell ta_in (trained ta_in + ta_lk; ta_lk is removed by '
                     'the 30 s training cap, see the module docstring)'),
    ('ur_pk',  'ur', 'Urdu - study cell ur_pk (trained ur_pk + ur_in)'),
    ('am_et',  'am', 'Amharic - study cell am_et'),
    ('crs_sc', 'crs', 'Seychellois Creole - unseen by both Whisper and TinyAya, the OOD probe'),
    # NOTE: crs_sc is the one entry that does NOT come from disco-eth. See DATASET_PATH_OVERRIDE.
    # ha_ng is deliberately absent: the study ALREADY evaluates it on
    # disco-eth/WorldSpeech test, so its in-domain config would duplicate an existing cell.
]

# Non-space-delimited scripts need CER rather than WER, matching create_yamls_fleurs_full.py's
# CER_LANG_IDS rule. None of the languages above is in that set, but the branch is kept so a
# later addition does not silently get a meaningless WER.
CER_LANGS = {'th', 'my', 'km', 'zh', 'ja', 'lo'}

# crs_sc must come from the ERISLab mirror, not disco-eth. That mirror's splits carry the
# duration-consistency cleaning (samples whose decoded audio length disagrees with the corpus
# `duration` column by >=1 s are removed) that the Seychellois Creole data required, and the
# study's crs_sc runs train and evaluate on it. Pointing the in-domain eval at disco-eth
# instead would evaluate on uncleaned audio the model never trained against.
DATASET_PATH_OVERRIDE = {'crs_sc': ('ERISLab/WorldSpeech', 'test_clean')}


class QuotedStr(str):
    pass


def quoted_scalar(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")


yaml.add_representer(QuotedStr, quoted_scalar)

output_dir = 'configs/datasets/short_ml'
os.makedirs(output_dir, exist_ok=True)

n_written = 0
for ws_config, lang_code, note in WORLDSPEECH_ENTRIES:
    filename = f'worldspeech_{ws_config}_test.yaml'
    filepath = os.path.join(output_dir, filename)
    ds_path, split = DATASET_PATH_OVERRIDE.get(ws_config,
                                               ('disco-eth/WorldSpeech', 'test'))
    yaml_data = {
        'dataset_path': QuotedStr(ds_path),
        'dataset': QuotedStr(ws_config),
        'split': QuotedStr(split),
        'force_asr_language': QuotedStr(lang_code),
        'eval_metrics': ['cer'] if lang_code in CER_LANGS else ['wer_all'],
    }
    if lang_code == 'en':
        yaml_data['norm_english'] = True

    with open(filepath, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    n_written += 1
    print(f'Created: {filename}  # {note}')

print(f'Successfully generated {n_written} config files.')
