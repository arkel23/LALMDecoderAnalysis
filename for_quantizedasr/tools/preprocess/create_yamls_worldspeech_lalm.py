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

# ALL WorldSpeech variants of every study language, not just the one each cell was trained on.
#
# An earlier version emitted one config per cell -- ta_in but not ta_lk, ur_pk but not ur_in --
# on the reasoning that the eval should match the reported cell. That under-used the corpus for
# no saving: these are evaluation-only configs, and WorldSpeech ships several country variants
# per language. Covering them all turns a single in-domain number into a dialect/accent
# generalisation axis at pure inference cost.
#
# Three statuses, carried in the manifest CSV rather than in the YAML (which stays byte-identical
# in shape to QuantizedASR's own generators):
#
#   in_training     the cell trained on this config. In-distribution.
#   dropped_by_cap  the training config LISTED it, but `max_input_length: 30` with a strict `<`
#                   removed every clip, so the model never saw it. This is `ta_lk`, and it is
#                   the most interesting entry here: evaluating on it is genuine HELD-OUT
#                   dialect transfer for a model whose config claims to have trained on it.
#   held_out        never in the training mix. Zero-shot accent/dialect transfer.
#
# The held_out English (7), Spanish (8) and French (2) variants are the accent-robustness axis
# that docs/EVAL_DATASET_PLAN.md flagged as missing and proposed adding EdAcc for -- available
# here in-domain, on the training corpus, for free.
#
# The variant list below was enumerated live from the Hub and then cross-checked against
# QuantizedASR's own `configs/train/worldspeech_llama_questions.yaml`, whose 120-entry
# `dataset_train` list is the frozen all-variants roster. The two agree exactly for every study
# language, so the list here is not a guess about what WorldSpeech ships.

# (WorldSpeech config, force_asr_language, study cell, status, note)
WORLDSPEECH_ENTRIES = [
    # --- English: trained on en_us, 7 further accents held out ----------------------
    ('en_us',  'en',  'en_us',  'in_training',   'trained variety'),
    ('en_au',  'en',  'en_us',  'held_out',      'Australian'),
    ('en_jm',  'en',  'en_us',  'held_out',      'Jamaican'),
    ('en_ke',  'en',  'en_us',  'held_out',      'Kenyan'),
    ('en_nz',  'en',  'en_us',  'held_out',      'New Zealand'),
    ('en_pk',  'en',  'en_us',  'held_out',      'Pakistani'),
    ('en_sl',  'en',  'en_us',  'held_out',      'Sierra Leonean'),
    ('en_zm',  'en',  'en_us',  'held_out',      'Zambian'),

    # --- Spanish: trained on es_mx. es_es is the superseded training variety --------
    ('es_mx',  'es',  'es_419', 'in_training',   'trained variety'),
    ('es_es',  'es',  'es_419', 'held_out',      'Spain -- the superseded training variety'),
    ('es_ar',  'es',  'es_419', 'held_out',      'Argentine'),
    ('es_cl',  'es',  'es_419', 'held_out',      'Chilean'),
    ('es_co',  'es',  'es_419', 'held_out',      'Colombian'),
    ('es_pe',  'es',  'es_419', 'held_out',      'Peruvian'),
    ('es_pr',  'es',  'es_419', 'held_out',      'Puerto Rican'),
    ('es_py',  'es',  'es_419', 'held_out',      'Paraguayan'),
    ('es_uy',  'es',  'es_419', 'held_out',      'Uruguayan'),

    # --- French: trained on fr_ca; two African varieties held out -------------------
    ('fr_ca',  'fr',  'fr_fr',  'in_training',   'trained variety'),
    ('fr_cd',  'fr',  'fr_fr',  'held_out',      'DR Congo'),
    ('fr_ci',  'fr',  'fr_fr',  'held_out',      "Cote d'Ivoire"),

    # --- Multi-config cells: BOTH varieties were trained on ------------------------
    ('ha_ng',  'ha',  'ha_ng',  'in_training',   'Nigeria'),
    ('ha_td',  'ha',  'ha_ng',  'in_training',   'Chad'),
    ('sw_ke',  'sw',  'sw_ke',  'in_training',   'Kenya'),
    ('sw_tz',  'sw',  'sw_ke',  'in_training',   'Tanzania'),
    ('ur_pk',  'ur',  'ur_pk',  'in_training',   'Pakistan'),
    ('ur_in',  'ur',  'ur_pk',  'in_training',   'India'),

    # --- Tamil: the config lists ta_lk, but the 30 s cap removed all of it ---------
    ('ta_in',  'ta',  'ta_in',  'in_training',   'India'),
    ('ta_lk',  'ta',  'ta_in',  'dropped_by_cap',
     'Sri Lanka -- listed in the training config but every clip is exactly 30.00 s, so the '
     'strict `< 30` filter removed all 23,261. Evaluating here is held-out dialect transfer.'),

    # --- Single-variety cells ------------------------------------------------------
    ('hi_in',  'hi',  'hi_in',  'in_training',   'only Hindi variety'),
    ('id_id',  'id',  'id_id',  'in_training',   'only Indonesian variety'),
    ('mr_in',  'mr',  'mr_in',  'in_training',   'only Marathi variety'),
    ('am_et',  'am',  'am_et',  'in_training',   'only Amharic variety'),
    ('crs_sc', 'crs', 'crs_sc', 'in_training',   'only Seychellois Creole variety'),
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

manifest = []
n_written = 0
for ws_config, lang_code, study_cell, status, note in WORLDSPEECH_ENTRIES:
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
    manifest.append({'config_file': filename, 'dataset_path': ds_path,
                     'dataset': ws_config, 'split': split, 'study_cell': study_cell,
                     'status': status, 'force_asr_language': lang_code, 'note': note})
    n_written += 1
    print(f'Created: {filename}  # [{status}] {note}')

# The manifest is how the analysis knows which evals are in-distribution, which are zero-shot
# accent transfer, and which is the cap-dropped Tamil variety. Keeping it out of the YAML means
# the configs stay exactly the shape QuantizedASR's other generators produce.
import csv
manifest_path = os.path.join(output_dir, 'worldspeech_lalm_manifest.csv')
with open(manifest_path, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
    w.writeheader()
    w.writerows(manifest)

print(f'\nSuccessfully generated {n_written} config files.')
print(f'Wrote manifest to {manifest_path}')
from collections import Counter
for status, n in sorted(Counter(m['status'] for m in manifest).items()):
    print(f'  {status:15s} {n}')
