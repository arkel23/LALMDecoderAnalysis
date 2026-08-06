"""Data dicts and the preprocessing pipeline for LALMDecoderAnalysis."""
import os
import re

import numpy as np
import pandas as pd

# --- Serials ----------------------------------------------------------------------------
# A serial tags a group of wandb runs sharing a configuration; SERIAL_DIC gives its label.
SERIAL_DIC = {
    0: 'Connector-only SFT',   # --freeze_encoder --freeze_decoder (SLAM-style)
    1: 'Superseded grid re-run',
    2: 'Control arms',
    3: 'Superseded control',
    4: 'Same-seed replicate',
    5: 'Superseded es_es condition',
}

# What each serial holds. Serial 0 is the analysis population: exactly 12 languages x the 4
# grid-wide variants, so a cross-language aggregate over it is correct without further filtering.
# Everything that is not a grid cell lives elsewhere, which is what makes that structural.
SERIAL_ROLE = {
    0: 'grid',                  # 48 runs, 12 languages x {earth, fire, global, water}
    1: 'superseded_grid',       # am_et and crs_sc originals, seed 42, replaced by seed 420
    2: 'control',               # base and qwen3-4b, crs_sc and ta_in only
    3: 'superseded_control',    # crs_sc/base original
    4: 'same_seed_replicate',   # en_us/water; the re-run set no new seed
    5: 'superseded_condition',  # trained on es_es, superseded by es_mx
}

GRID_SERIAL = 0
# t1 spans these so the crs_sc cell keeps all six models -- the OOD probe is the one place a
# control arm belongs beside the grid.
CURVE_SERIALS = (0, 2)

# (canonical, superseded). Pairing only 0<->1 would drop the control replicate and the
# same-seed en_us/water pair, which is the outlier the noise-floor argument rests on.
REPLICATE_SERIAL_PAIRS = ((0, 1), (2, 3), (0, 4))

SERIALS_EXPLANATIONS = []

SETTINGS_DIC = {}


def get_canonical_labels(present=None):
    """Legend/hue order. De-duplicated because a duplicate in hue_order reassigns one
    category's colour while positions and labels stay pixel-identical."""
    labels = list(SERIAL_DIC.values())
    if present is not None:
        labels = [l for l in labels if l in set(present)]
    seen, out = set(), []
    for l in labels:
        if l not in seen:
            seen.add(l)
            out.append(l)
    return out


# The variants differ ONLY in post-training data composition -- the premise the comparison
# rests on, and still unverified against the Tiny Aya report.
METHODS_DIC = {
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-base': 'TinyAya-base',
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-global': 'TinyAya-global',
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-earth': 'TinyAya-earth',
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-fire': 'TinyAya-fire',
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-water': 'TinyAya-water',
    'q2a_openai/whisper-medium_Qwen/Qwen3-4B': 'Qwen3-4B',
}

# Short label for tables that group by decoder rather than by the full model_id.
MODEL_SHORT = {
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-base': 'base',
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-global': 'global',
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-earth': 'earth',
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-fire': 'fire',
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-water': 'water',
    'q2a_openai/whisper-medium_Qwen/Qwen3-4B': 'qwen3-4b',
}

# base and Qwen3-4B exist only for crs_sc, so cross-language aggregates must restrict to
# these four or compare a 10-language mean against a 1-language mean.
CORE_VARIANTS = ('earth', 'fire', 'global', 'water')

# Qwen3-4B is the non-Aya control: a topline reference row, never in the controlled test.
MODEL_FAMILY = {
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-base': 'Non-regional',
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-global': 'Non-regional',
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-earth': 'Regional',
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-fire': 'Regional',
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-water': 'Regional',
    'q2a_openai/whisper-medium_Qwen/Qwen3-4B': 'Non-Aya control',
}

# Models excluded from aggregates. Keep the raw data; exclude at analysis time only.
EXCLUDED_MODELS_AGGREGATE = [
    # Empty deliberately: en_us/water replicated, so it is an effect, not a failed run, and
    # it is the grid's strongest against-hypothesis point. The two runs differ by ~5 CER,
    # so water-on-English is both bad AND unstable.
]


def is_excluded_from_aggregate(model_id, dataset):
    return any(e['model_id'] == model_id and e['dataset'] == dataset
               for e in EXCLUDED_MODELS_AGGREGATE)


# --- Languages --------------------------------------------------------------------------
# TinyAya regions per the report (arXiv:2603.11510, Sec 2.3.3). West Asia sits with Earth,
# not Water:  Earth = Africa + West Asia | Fire = South Asia | Water = Asia-Pacific + Europe
#
# crs_sc is None, not 'earth': African, but supported by neither Whisper nor TinyAya, so it
# is an OOD probe rather than a region-matched cell and is reported separately.
LANGUAGE_REGION = {
    'ha_ng': 'earth',
    'sw_ke': 'earth',
    'am_et': 'earth',      # Amharic -- African, confirmed in the report's Africa table
    'ur_pk': 'fire',       # Urdu -- see URDU_REGION_NOTE below
    'hi_in': 'fire',
    'mr_in': 'fire',
    'ta_in': 'fire',
    'en_us': 'water',
    'fr_fr': 'water',
    'es_419': 'water',
    'id_id': 'water',
    'crs_sc': None,
}

LANGUAGE_STATUS = {
    'crs_sc': 'ood_encoder_and_decoder',
}

# Urdu is ambiguous: report Table 1 says South Asia, Appendix Table 10 says West Asia.
# Assigned fire because the fire mix holds 3.4% Urdu against earth's 1.3%.
URDU_REGION_NOTE = ('report Table 1 says South Asia, Appendix Table 10 lists it under West '
                    'Asia; assigned fire because the fire mix has 3.4% Urdu vs 1.3% earth')

LANGUAGE_DIC = {
    'crs_sc': 'Seychellois Creole',
    'am_et': 'Amharic',
    'ur_pk': 'Urdu',
    'en_us': 'English',
    'es_419': 'Spanish (LatAm)',
    'fr_fr': 'French',
    'ha_ng': 'Hausa',
    'hi_in': 'Hindi',
    'id_id': 'Indonesian',
    'mr_in': 'Marathi',
    'sw_ke': 'Swahili',
    'ta_in': 'Tamil',
}

# Cells whose training condition is not a clean match for their eval condition. Not errors,
# but not interchangeable with the clean cells either.
#
# Multi-config interleaving is NOT a confound: 'all_exhausted_without_replacement' makes the
# stream exactly the sum of its parts. Proven by verify_interleave_semantics.py.
TRAIN_EVAL_MATCH = {
    'fr_fr': 'dialect_mismatch',      # trains fr_ca (Canadian), evaluates fr_fr (European)
    # es_419 was listed here while it trained on es_es (Spain). Those runs were deleted on
    # 2026-08-01 and re-run from es_mx, which is inside the Latin American variety group
    # FLEURS es_419 evaluates -- so it is no longer a mismatch and must not be flagged as one.
}

# Languages whose training stream is built from more than one WorldSpeech config. Recorded
# because it changes what the published per-config hour count means (it becomes a lower bound
# on the combined stream), not because the interleaving distorts sampling.
MULTI_CONFIG_TRAIN = {
    'ur_pk': ('ur_pk', 'ur_in'),
    'ha_ng': ('ha_ng', 'ha_td'),
    'sw_ke': ('sw_ke', 'sw_tz'),
}

# --- Training stream composition ------------------------------------------------------
# The (dataset_path, configs, split) each language's connector was trained on, from the
# upstream configs/train/*ws*.yaml. Used to look up authoritative example counts.
TRAIN_CONFIGS = {
    'en_us':  ('disco-eth/WorldSpeech', ('en_us',),          'train'),
    'fr_fr':  ('disco-eth/WorldSpeech', ('fr_ca',),          'train'),
    # es_419 trains on es_mx, NOT es_es. The Spain-Spanish runs were deleted 2026-08-01 and
    # re-run from Mexican Spanish, which also removes the dialect mismatch this cell used to
    # carry (Mexican Spanish is within the Latin American variety FLEURS es_419 evaluates).
    'es_419': ('disco-eth/WorldSpeech', ('es_mx',),          'train'),
    'am_et':  ('disco-eth/WorldSpeech', ('am_et',),          'train'),
    'ur_pk':  ('disco-eth/WorldSpeech', ('ur_pk', 'ur_in'),  'train'),
    'hi_in':  ('disco-eth/WorldSpeech', ('hi_in',),          'train'),
    'id_id':  ('disco-eth/WorldSpeech', ('id_id',),          'train'),
    'mr_in':  ('disco-eth/WorldSpeech', ('mr_in',),          'train'),
    'sw_ke':  ('disco-eth/WorldSpeech', ('sw_ke', 'sw_tz'),  'train'),
    'ta_in':  ('disco-eth/WorldSpeech', ('ta_in',),          'train'),
    'ha_ng':  ('disco-eth/WorldSpeech', ('ha_ng', 'ha_td'),  'train'),
    'crs_sc': ('ERISLab/WorldSpeech',   ('crs_sc',),         'train_val_exc_clean'),
}

# --- WorldSpeech example counts: FROZEN SNAPSHOT ----------------------------------------
# num_examples per training config, from the HF dataset builder metadata (2026-07-30).
# Regenerate with verify_dataset_durations.py.
WORLDSPEECH_TRAIN_EXAMPLES = {
    'en_us':  666718,
    'fr_ca':  207449,
    'es_es':  866048,
    'es_mx':  205972,
    'hi_in':  577382,
    'sw_ke':  101774,
    'sw_tz':  200314,
    'ha_ng':  11865,
    'ha_td':  15390,
    'ta_in':  8846,
    'mr_in':  58201,
    'id_id':  101112,
    'am_et':  8873,
    'ur_pk':  28142,
    'ur_in':  2937,
}


# --- The duration cap -------------------------------------------------------------------
# The upstream filter keeps a clip when `length < max_input_length` -- STRICT -- so a clip at
# exactly 30.000 s is dropped.
MAX_INPUT_LENGTH_S = 30

# Fraction of each config's clips at or above the cap. FROZEN SNAPSHOT (2026-07-30, 100-row
# `duration` sample). fr_ca loses ~4%, the rest ~0.
CONFIG_DURATION_AT_CAP = {
    'ta_in': 0.00,
    'fr_ca': 0.04,
    'hi_in': 0.00,
    'sw_ke': 0.00,
    'ha_ng': 0.00,
    'en_us': 0.00,
    'es_es': 0.00,
    'es_mx': 0.00,
    'sw_tz': 0.00,
    # ta_in, ha_td, mr_in, id_id: the rows endpoint returned HTTP 500 (config not cached), but
    # each language's stream reconciles with its full example count in t4, which is independent
    # evidence that the cap is not removing a material share.
}


# Configs known to sit entirely at the cap, acknowledged rather than failed on, so the screen
# gates on NEW regressions. Empty: the one such config is no longer part of any training stream.
KNOWN_AT_CAP_CONFIGS = ()


def expected_stream_examples(language, post_filter=False):
    """Total examples in a language's training stream; None when any part is unknown.

    Summing is valid because interleaving is lossless (verify_interleave_semantics.py)."""
    entry = TRAIN_CONFIGS.get(language)
    if not entry:
        return None
    _, configs, _ = entry
    counts = [WORLDSPEECH_TRAIN_EXAMPLES.get(c) for c in configs]
    if not all(c is not None for c in counts):
        return None
    if not post_filter:
        return sum(counts)
    # Post-filter: subtract each config's at-cap share. This is what the model actually saw,
    # and it is what makes ta_in reconcile (32,107 -> 8,846) instead of looking anomalous.
    return sum(c * (1.0 - CONFIG_DURATION_AT_CAP.get(cfg, 0.0))
               for cfg, c in zip(configs, counts))


# --- Resource tiers: the study's independent variable -----------------------------------
# Post-filter training hours from two sources, kept separate because they disagree for mr_in
# (~217 h computed against a stated ~110) and that disagreement should stay visible.
LANGUAGE_HOURS_COMPUTED = {
    'en_us': 3510.2,
    'hi_in': 2480.4,
    'fr_fr': 983.6,
    'es_419': 855.3,     # es_mx: 205,972 clips x 14.95 s
    'sw_ke': 826.0,
    'ur_pk': 65.4,       # ur_pk only; ur_in's duration endpoint was uncached
    'am_et': 37.9,
    # id_id, mr_in, ta_in, ha_ng, crs_sc: duration endpoint uncached for at least one config
}

MR_IN_HOURS_DISCREPANCY = (
    'mr_in: 58,201 clips x 13.45 s (the mean the training logs imply) is ~217 h, against a '
    'stated ~110-120 h. Unresolved; the tier below uses the stated figure.')

# Tier labels from the project's accounting. The whole point of adding am_et and ur_pk was to
# populate the middle and low tiers, so this is the study's independent variable.
RESOURCE_TIER = {
    'en_us': 'high', 'fr_fr': 'high', 'es_419': 'high', 'hi_in': 'high',
    'id_id': 'high', 'sw_ke': 'high', 'crs_sc': 'high',      # all >200 h
    'ha_ng': 'mid', 'mr_in': 'mid',                          # ~110-120 h
    'ur_pk': 'low',                                          # ~80 h
    'am_et': 'very_low', 'ta_in': 'very_low',                # ~40 h
}

TIER_ORDER = ('very_low', 'low', 'mid', 'high')

# --- The two axes the eval sets differ on -----------------------------------------------
# A large train-eval loss gap can mean overfitting OR a different domain/accent. Training is
# always WorldSpeech, so a FLEURS eval is cross-domain -- 10 of 12 languages.
EVAL_DOMAIN = {
    'ha_ng': 'in_domain',        # disco-eth/WorldSpeech test
    'crs_sc': 'in_domain',       # ERISLab/WorldSpeech val_clean
}   # every other language: cross_domain (google/fleurs validation)

# ACCENT_MATCH: whether the training config's variety matches the evaluated variety.
ACCENT_MATCH = {
    'fr_fr': 'different',        # trains fr_ca (Canadian), evaluates fr_fr (European)
    'es_419': 'related',         # trains es_mx, evaluates es_419 -- both Latin American
    'ur_pk': 'partial',          # trains ur_pk + ur_in, evaluates ur_pk
    'sw_ke': 'partial',          # trains sw_ke + sw_tz, evaluates sw_ke
    'ha_ng': 'partial',          # trains ha_ng + ha_td, evaluates ha_ng
}   # every other language: 'same'

# The split each cell's best checkpoint was selected on. An eval on that same split is not
# held out. Only ha_ng collides with what the sweeps evaluate; ha_td is its substitute.
SELECTION_SPLIT = {
    'am_et': ('google/fleurs', 'am_et', 'validation'),
    'crs_sc': ('ERISLab/WorldSpeech', 'crs_sc', 'val_clean'),
    'en_us': ('google/fleurs', 'en_us', 'validation'),
    'es_419': ('google/fleurs', 'es_419', 'validation'),
    'fr_fr': ('google/fleurs', 'fr_fr', 'validation'),
    'ha_ng': ('disco-eth/WorldSpeech', 'ha_ng', 'test'),
    'hi_in': ('google/fleurs', 'hi_in', 'validation'),
    'id_id': ('google/fleurs', 'id_id', 'validation'),
    'mr_in': ('google/fleurs', 'mr_in', 'validation'),
    'sw_ke': ('google/fleurs', 'sw_ke', 'validation'),
    'ta_in': ('google/fleurs', 'ta_in', 'validation'),
    'ur_pk': ('google/fleurs', 'ur_pk', 'validation'),
}

# In-domain eval configs that reuse their own cell's selection split. Prefer the substitute.
REUSES_SELECTION_SPLIT = {'ha_ng': 'ha_td'}

# The one WorldSpeech config that is each cell's in-domain point. All 33 evaluated variants
# normalise to the same study cell, so without choosing, a trained variety would be averaged
# with zero-shot accent transfer. The trained variety everywhere except ha_ng -> ha_td.
IN_DOMAIN_PRIMARY = {
    'am_et': 'am_et', 'crs_sc': 'crs_sc', 'en_us': 'en_us', 'es_419': 'es_mx',
    'fr_fr': 'fr_ca', 'ha_ng': 'ha_td', 'hi_in': 'hi_in', 'id_id': 'id_id',
    'mr_in': 'mr_in', 'sw_ke': 'sw_ke', 'ta_in': 'ta_in', 'ur_pk': 'ur_pk',
}


def in_domain_role(study_cell, dataset, eval_domain):
    """'primary' for the cell's in-domain point, 'accent_transfer' for its other varieties."""
    if eval_domain != 'in_domain':
        return ''
    return 'primary' if IN_DOMAIN_PRIMARY.get(study_cell) == dataset else 'accent_transfer'


def is_selection_split(language, dataset_path, dataset, split):
    """True when this eval is on the split the cell's checkpoint was chosen on."""
    return SELECTION_SPLIT.get(language) == (dataset_path, dataset, split)


# WorldSpeech eval configs are named after the TRAINING variety (fr_ca, es_mx), while the study
# cell is named after the FLEURS variety (fr_fr, es_419). Without this map those rows get no
# resource tier and no region, and silently drop out of any grouped table.
TRAIN_CONFIG_TO_CELL = {
    'fr_ca': 'fr_fr',
    'ta_lk': 'ta_in',
    'es_mx': 'es_419',
    'es_es': 'es_419',
    'ur_in': 'ur_pk',
    'sw_tz': 'sw_ke',
    'ha_td': 'ha_ng',
    # The held-out variants the sweeps evaluate. Without these they carry no language_name and
    # no resource_tier, so the accent-transfer axis cannot be read per language.
    'en_au': 'en_us', 'en_jm': 'en_us', 'en_ke': 'en_us', 'en_nz': 'en_us',
    'en_pk': 'en_us', 'en_sl': 'en_us', 'en_zm': 'en_us',
    'es_ar': 'es_419', 'es_cl': 'es_419', 'es_co': 'es_419', 'es_pe': 'es_419',
    'es_pr': 'es_419', 'es_py': 'es_419', 'es_uy': 'es_419',
    'fr_cd': 'fr_fr', 'fr_ci': 'fr_fr',
}


# A trained checkpoint's model_id names the checkpoint, not the model:
# ERISLab/q2a_openai_whisper-medium_CohereLabs_tiny-aya-water_ws-ha_ng-700. Serial 11 therefore
# joins to nothing until the id is reduced to its parent, which is a METHODS_DIC key.
CHECKPOINT_RE = re.compile(
    r'^[\w.-]+/q2a_(?P<enc_org>[^_]+)_(?P<enc>[\w.-]+?)_(?P<dec_org>[^_]+)_(?P<dec>[\w.-]+?)'
    r'_(?P<corpus>[a-z]+)-(?P<lang>[a-z]{2,3}_[a-z]{2,3})-(?P<step>\d+)$')


def parent_model_id(model_id):
    """Checkpoint id -> the METHODS_DIC key for the model it fine-tuned.

    Returns the id unchanged when it is already a parent, so this is safe to map over a column
    holding both.
    """
    m = CHECKPOINT_RE.match(str(model_id))
    if not m:
        return model_id
    g = m.groupdict()
    return f"q2a_{g['enc_org']}/{g['enc']}_{g['dec_org']}/{g['dec']}"


def checkpoint_fields(model_id):
    """-> (training language, step) for a checkpoint id; (None, None) for a parent id."""
    m = CHECKPOINT_RE.match(str(model_id))
    return (m.group('lang'), int(m.group('step'))) if m else (None, None)


def to_study_cell(code):
    """Normalise a dataset config name to the study's language cell."""
    return TRAIN_CONFIG_TO_CELL.get(code, code)


def get_eval_domain(language):
    return EVAL_DOMAIN.get(language, 'cross_domain')


def get_accent_match(language):
    return ACCENT_MATCH.get(language, 'same')


# --- Tiny Aya post-training composition -------------------------------------------------
# Per-language share of each variant's post-training mix (report Appendix A), which turns
# "specialisation" from a label into a continuous variable. From fetch_tinyaya_composition.py.
TINYAYA_COMPOSITION_CSV = os.path.join('data', 'tinyaya_report',
                                       'tinyaya_language_composition_wide.csv')

# The report names languages in English; our cells are FLEURS locale codes.
LANGUAGE_TO_REPORT_NAME = {
    'en_us': 'English', 'fr_fr': 'French', 'es_419': 'Spanish', 'id_id': 'Indonesian',
    'hi_in': 'Hindi', 'mr_in': 'Marathi', 'ta_in': 'Tamil', 'ur_pk': 'Urdu',
    'am_et': 'Amharic', 'ha_ng': 'Hausa', 'sw_ke': 'Swahili',
    # crs_sc (Seychellois Creole) is absent from the report by construction: it is not one of
    # Tiny Aya's 70 languages, which is exactly what makes it the OOD probe.
}


def load_tinyaya_composition(path=None):
    """Wide per-language variant exposure. None when the CSV has not been generated yet."""
    path = path or TINYAYA_COMPOSITION_CSV
    if not os.path.exists(path):
        return None
    comp = pd.read_csv(path)
    name_to_lang = {v: k for k, v in LANGUAGE_TO_REPORT_NAME.items()}
    comp = comp[comp['language'].isin(name_to_lang)].copy()
    comp['dataset'] = comp['language'].map(name_to_lang)
    return comp.rename(columns={'language': 'report_language'})


# --- Datasets and metrics ---------------------------------------------------------------
# Keyed on the dataset_name standarize_df builds: dataset_path_dataset_split. The eval sets
# differ per language, so raw CER must never be pooled across languages.
DATASETS_DIC = {
    'ERISLab/WorldSpeech_crs_sc_val_clean': 'Seychellois Creole (WS)',
    'disco-eth/WorldSpeech_ha_ng_test': 'Hausa (WS)',
    'google/fleurs_en_us_validation': 'English (FLEURS)',
    'google/fleurs_es_419_validation': 'Spanish (FLEURS)',
    'google/fleurs_fr_fr_validation': 'French (FLEURS)',
    'google/fleurs_hi_in_validation': 'Hindi (FLEURS)',
    'google/fleurs_id_id_validation': 'Indonesian (FLEURS)',
    'google/fleurs_mr_in_validation': 'Marathi (FLEURS)',
    'google/fleurs_sw_ke_validation': 'Swahili (FLEURS)',
    'google/fleurs_ta_in_validation': 'Tamil (FLEURS)',
}

METRIC_DIC = {
    'acc': 'Accuracy',
    'wer': 'WER',
    'cer': 'CER',
}

VAR_DIC = {
    'dataset_name': 'Dataset',
    'model_id': 'Decoder variant',
    'no_params': 'Number of Parameters (10^6)',
    'serial': 'Condition',
}


def half_up(value, decimals):
    """Round half away from zero. Built-in round() is banker's: round(5.25, 1) is 5.2."""
    from decimal import Decimal, ROUND_HALF_UP
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    quant = Decimal(1).scaleb(-decimals)
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


def assert_unique_keys(df, key_cols, label=''):
    """Fail loudly if key_cols does not uniquely identify a row.

    A merge on a non-unique key silently yields the k^2 cross product, which looks only like
    a larger n. Every merge and paired subtraction here calls this first."""
    dupes = df.duplicated(subset=key_cols, keep=False)
    if dupes.any():
        offenders = df.loc[dupes, key_cols].drop_duplicates()
        raise AssertionError(
            f'{label or "df"}: {key_cols} is not unique -- '
            f'{int(dupes.sum())} duplicated rows across {len(offenders)} key(s):\n'
            f'{offenders.to_string(index=False)}'
        )
    return df


def add_language_columns(df, lang_col='dataset'):
    """Attach region / status / train-eval-match metadata keyed on the language code."""
    if lang_col not in df.columns:
        return df
    df = df.copy()
    df['language_name'] = df[lang_col].map(LANGUAGE_DIC)
    df['region'] = df[lang_col].map(LANGUAGE_REGION)
    df['language_status'] = df[lang_col].map(LANGUAGE_STATUS).fillna('in_domain')
    df['train_eval_match'] = df[lang_col].map(TRAIN_EVAL_MATCH).fillna('clean')
    if 'model_id' in df.columns:
        df['model_short'] = df['model_id'].map(MODEL_SHORT)
    return df


def rename_var(x):
    for d in (SETTINGS_DIC, METHODS_DIC, DATASETS_DIC, VAR_DIC, SERIAL_DIC, METRIC_DIC):
        if x in d:
            return d[x]
    return x


def rename_vars(df, var_rename=False, args=None):
    for col in ('setting', 'model_id', 'model_name_extractor', 'dataset_name', 'family',
                'serial'):
        if col in df.columns:
            df[col] = df[col].apply(rename_var)

    if var_rename:
        df.rename(columns=VAR_DIC, inplace=True)
        for k, v in VAR_DIC.items():
            for attr in ('x_var_name', 'y_var_name', 'hue_var_name', 'style_var_name',
                         'size_var_name'):
                if getattr(args, attr, None) == k:
                    setattr(args, attr, v)
    return df


def add_setting(df):
    if SERIALS_EXPLANATIONS:
        conditions = [(df['serial'] == s) for s in SERIAL_DIC]
        df['setting'] = np.select(conditions, SERIALS_EXPLANATIONS, default='')
    else:
        df['setting'] = ''
    return df


def standarize_df(df):
    df = df.fillna({'dataset_path': '', 'dataset': '', 'split': ''})
    df['dataset_name'] = (
        df['dataset_path'].astype(str).str.strip() + '_' +
        df['dataset'].astype(str).str.strip() + '_' +
        df['split'].astype(str).str.strip()
    )
    return add_setting(df)


def keep_columns(df):
    kw_list = ['acc', 'wer', 'cer', 'rtfx', 'force_asr_language', 'task']
    keep = ['dataset_path', 'dataset_name', 'dataset', 'split', 'model_id', 'serial',
            'setting', 'no_params'] + \
        [c for c in df.columns if any(kw in c for kw in kw_list)]
    return df[[c for c in dict.fromkeys(keep) if c in df.columns]]


def filter_df(df, keep_datasets=None, keep_methods=None, keep_serials=None,
              filter_datasets=None, filter_methods=None, filter_serials=None,
              keep_ratios=None, keep_extractors=None):
    if keep_datasets:
        df = df[df['dataset_name'].isin(keep_datasets)]
    if keep_methods:
        df = df[df['model_id'].isin(keep_methods)]
    if keep_extractors:
        df = df[df['model_name_extractor'].isin(keep_extractors)]
    if keep_serials:
        df = df[df['serial'].isin(keep_serials)]
    if filter_datasets:
        df = df[~df['dataset_name'].isin(filter_datasets)]
    if filter_methods:
        df = df[~df['model_id'].isin(filter_methods)]
    if filter_serials:
        df = df[~df['serial'].isin(filter_serials)]
    return df


def extra_columns(df):
    if 'model_id' in df.columns:
        df['model_family'] = df['model_id'].map(MODEL_FAMILY)
    return df


def preprocess_df(df, type='acc', keep_datasets=None, keep_methods=None, keep_serials=None,
                  filter_datasets=None, filter_methods=None, filter_serials=None,
                  keep_ratios=None, keep_extractors=None):
    df = standarize_df(df)
    df = keep_columns(df)
    df = extra_columns(df)
    df = filter_df(df, keep_datasets, keep_methods, keep_serials,
                   filter_datasets, filter_methods, filter_serials, keep_ratios,
                   keep_extractors)
    return sort_df(df, method_only=True)


def drop_na(df, args):
    subset = [args.x_var_name, args.y_var_name]
    for attr in ('hue_var_name', 'style_var_name', 'size_var_name'):
        if getattr(args, attr, None):
            subset.append(getattr(args, attr))
    return df.dropna(subset=[c for c in subset if c in df.columns])


def sort_df(df, method_only=False, dataset_only=False, raw_data=False):
    if 'model_id' in df.columns:
        df['method_order'] = pd.Categorical(df['model_id'], categories=METHODS_DIC.keys(),
                                            ordered=True)
    if 'dataset_name' in df.columns and DATASETS_DIC:
        df['dataset_order'] = pd.Categorical(df['dataset_name'],
                                             categories=DATASETS_DIC.keys(), ordered=True)
    by = [c for c in ('serial', 'setting', 'dataset_order', 'method_order')
          if c in df.columns]
    if by:
        df = df.sort_values(by=by, ascending=True)
    return df.drop(columns=[c for c in ('method_order', 'dataset_order')
                            if c in df.columns])
