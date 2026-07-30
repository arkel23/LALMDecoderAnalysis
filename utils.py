"""
Data dicts and the preprocessing pipeline for LALMDecoderAnalysis.

Pruned from MultilingualQASR/utils.py per the "duplicate, don't share" convention: the
generic plumbing (rename/filter/sort/preprocess) is kept, and everything specific to that
paper is gone -- the ~100-language hours table, the resource-tier binning
(get_resource_bin), the LANGUAGE_* dicts, and the NEEDS_CER-driven primary_error_rate
switch. None of that applies to a decoder-SFT comparison.

>>> EVERY DICT BELOW IS A PLACEHOLDER. Fill them from the real wandb runs before trusting
>>> any plot. In particular METHODS_DIC, SERIAL_DIC and DATASETS_DIC currently describe the
>>> intended TinyAya matrix, not observed data. See HANDOVER.md.
"""
import numpy as np
import pandas as pd

# --- Serials: one per experimental condition. Fill in as runs are logged. ------------
# Convention inherited from the sibling repos: a serial is an integer tag on a group of
# wandb runs sharing a training/eval configuration, and SERIAL_DIC maps it to the label
# used in figure legends.
SERIAL_DIC = {
    0: 'Connector-only SFT',   # --freeze_encoder --freeze_decoder (SLAM-style)
}

SERIALS_EXPLANATIONS = []

SETTINGS_DIC = {}


def get_canonical_labels(present=None):
    """Legend/hue order, de-duplicated.

    The de-duplication is deliberate and load-bearing: a hue_order list containing a
    duplicate silently corrupts seaborn's palette assignment (one category is drawn in
    another's colour) while box positions and labels stay pixel-identical, so a
    content-only check will not catch it. This bug was found for real in a sibling repo.
    """
    labels = list(SERIAL_DIC.values())
    if present is not None:
        labels = [l for l in labels if l in set(present)]
    seen, out = set(), []
    for l in labels:
        if l not in seen:
            seen.add(l)
            out.append(l)
    return out


# --- The TinyAya decoder variants under comparison ----------------------------------
# The whole point of the project: these differ ONLY in the data composition used for their
# post-training. Verify that claim against the official Tiny Aya report before drawing a
# causal conclusion (see HANDOVER.md, "Correctness risks").
# NOTE: the seeded keys said 'q2a_whisper_small_tiny_aya_*'. Those match nothing -- the runs
# use whisper-MEDIUM and the full HF paths below. Verified against wandb serial 0.
METHODS_DIC = {
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-base': 'TinyAya-base',
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-global': 'TinyAya-global',
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-earth': 'TinyAya-earth',
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-fire': 'TinyAya-fire',
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-water': 'TinyAya-water',
    'q2a_openai/whisper-medium_Qwen/Qwen3-4B': 'Qwen3-4B',
}

# Short label keyed on the variant suffix, for tables that group by decoder rather than
# by the full model_id string.
MODEL_SHORT = {
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-base': 'base',
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-global': 'global',
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-earth': 'earth',
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-fire': 'fire',
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-water': 'water',
    'q2a_openai/whisper-medium_Qwen/Qwen3-4B': 'qwen3-4b',
}

# The four variants that were run across the whole language grid. base and Qwen3-4B exist
# only for crs_sc, so any cross-language aggregate must be restricted to these four or it
# silently compares a 10-language mean against a 1-language mean.
CORE_VARIANTS = ('earth', 'fire', 'global', 'water')

# Grouping for figures that aggregate variants (e.g. regional vs non-regional).
# Qwen3-4B is not a TinyAya variant at all -- it is the non-Aya control, and it belongs in
# a topline reference row, never in the controlled comparison.
MODEL_FAMILY = {
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-base': 'Non-regional',
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-global': 'Non-regional',
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-earth': 'Regional',
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-fire': 'Regional',
    'q2a_openai/whisper-medium_CohereLabs/tiny-aya-water': 'Regional',
    'q2a_openai/whisper-medium_Qwen/Qwen3-4B': 'Non-Aya control',
}

# Models logged but excluded from aggregates, with the reason. Keep the raw data; exclude
# at analysis time only, and state the reason in the paper.
#
# The en_us/water run converges to ~20 CER while earth/fire/global reach 4.5-5.4 on the
# identical eval set. Its curve (225 -> 60 -> 33 -> 25 -> 21 -> 20) is a converged-but-bad
# optimisation, not a late spike, so it is a failed run rather than a decoder effect.
EXCLUDED_MODELS_AGGREGATE = [
    {
        'model_id': 'q2a_openai/whisper-medium_CohereLabs/tiny-aya-water',
        'dataset': 'en_us',
        'reason': 'optimisation failure: converges to ~20 CER vs 4.5-5.4 for the other '
                  'three variants on the identical eval set',
    },
]


def is_excluded_from_aggregate(model_id, dataset):
    return any(e['model_id'] == model_id and e['dataset'] == dataset
               for e in EXCLUDED_MODELS_AGGREGATE)


# --- Languages ----------------------------------------------------------------------
# TinyAya regional groupings. Earth = African, Fire = South Asian, Water = APAC / West
# Asia / Europe.
#
# crs_sc (Seychellois Creole) is deliberately None, NOT 'earth'. It is an African language,
# but it is officially supported by neither Whisper nor TinyAya, so it is the one cell where
# BOTH the encoder and the decoder are unseen. That makes it an out-of-distribution transfer
# probe rather than a region-matched cell, and including it in the matched-vs-mismatched test
# would be comparing a different thing. It is reported separately.
LANGUAGE_REGION = {
    'ha_ng': 'earth',
    'sw_ke': 'earth',
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

LANGUAGE_DIC = {
    'crs_sc': 'Seychellois Creole',
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

# Cells where the training condition is NOT a clean match for the eval condition. These are
# not errors to drop, but they are not interchangeable with the clean cells either, and a
# region-level claim that leans on them is weaker than it looks.
#
# NOTE on interleaving, which used to be listed here and is NOT a confound:
# ta_in+ta_lk, ha_ng+ha_td and sw_ke+sw_tz are loaded by interleave_datasets with uniform
# 1/N probabilities, which reads as though the smaller config would be oversampled. It is
# not. The strategy is 'all_exhausted_without_replacement', so an exhausted config is never
# recycled and the interleaved stream is exactly the sum of its parts, each example once.
# verify_interleave_semantics.py proves this empirically (125 == 100 + 25, no duplicates, in
# both the map-style and streaming paths). The uniform probabilities affect only ARRIVAL
# ORDER, not how many times an example is seen. Do not re-add this as a confound.
TRAIN_EVAL_MATCH = {
    'fr_fr': 'dialect_mismatch',      # trains fr_ca, evaluates fr_fr
    'es_419': 'dialect_mismatch',     # trains es_es (wandb says es_mx), evaluates es_419
}

# Languages whose training stream is built from more than one WorldSpeech config. Recorded
# because it changes what the published per-config hour count means (it becomes a lower bound
# on the combined stream), not because the interleaving distorts sampling.
MULTI_CONFIG_TRAIN = {
    'ta_in': ('ta_in', 'ta_lk'),
    'ha_ng': ('ha_ng', 'ha_td'),
    'sw_ke': ('sw_ke', 'sw_tz'),
}

# --- Training stream composition ------------------------------------------------------
# The (dataset_path, configs, split) each language's connector was trained on, from the
# upstream configs/train/*ws*.yaml. Used to look up authoritative example counts.
TRAIN_CONFIGS = {
    'en_us':  ('disco-eth/WorldSpeech', ('en_us',),          'train'),
    'fr_fr':  ('disco-eth/WorldSpeech', ('fr_ca',),          'train'),
    'es_419': ('disco-eth/WorldSpeech', ('es_es',),          'train'),
    'hi_in':  ('disco-eth/WorldSpeech', ('hi_in',),          'train'),
    'id_id':  ('disco-eth/WorldSpeech', ('id_id',),          'train'),
    'mr_in':  ('disco-eth/WorldSpeech', ('mr_in',),          'train'),
    'sw_ke':  ('disco-eth/WorldSpeech', ('sw_ke', 'sw_tz'),  'train'),
    'ta_in':  ('disco-eth/WorldSpeech', ('ta_in', 'ta_lk'),  'train'),
    'ha_ng':  ('disco-eth/WorldSpeech', ('ha_ng', 'ha_td'),  'train'),
    'crs_sc': ('ERISLab/WorldSpeech',   ('crs_sc',),         'train_val_exc_clean'),
}

# --- WorldSpeech example counts: FROZEN SNAPSHOT -------------------------------------
# num_examples per training config, read 2026-07-30 from the HuggingFace dataset builder
# metadata (authoritative, and cheap -- no audio downloaded). Regenerate with
# verify_dataset_durations.py.
#
# This replaced an earlier hours-based snapshot taken from a summarised web fetch of the
# WorldSpeech paper's table. That snapshot was the weakest input in the analysis and it
# produced a false positive: it implied Tamil should hold ~240 h, which disagreed with the
# reconstructed stream and was written up as a data-integrity problem. It was not one. Direct
# testing on the real data (verify_dataset_durations.py --load) showed the Tamil configs
# interleave losslessly and that the duration-consistency filter removes ZERO samples.
# Compare against example counts, which have real provenance -- not against numbers recovered
# from prose.
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
    'ta_lk':  23261,
    'mr_in':  58201,
    'id_id':  101112,
}


def expected_stream_examples(language):
    """Total examples in a language's training stream.

    Sound because interleaving is lossless: 'all_exhausted_without_replacement' never
    recycles an exhausted config, so a combined stream is exactly the sum of its parts
    (verify_interleave_semantics.py, and confirmed on the real Tamil configs).
    Returns None when any part is unknown -- crs_sc trains on an ERISLab mirror whose split
    is not in the snapshot.
    """
    entry = TRAIN_CONFIGS.get(language)
    if not entry:
        return None
    _, configs, _ = entry
    counts = [WORLDSPEECH_TRAIN_EXAMPLES.get(c) for c in configs]
    return sum(counts) if all(c is not None for c in counts) else None


# --- Datasets and metrics -----------------------------------------------------------
# Keyed on the dataset_name that standarize_df builds: dataset_path_dataset_split.
# Eight languages evaluate on FLEURS, one on WorldSpeech test, one on the ERISLab mirror --
# so raw CER must never be pooled across languages.
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
    """Round half away from zero, the way a printed number is expected to round.

    Python's built-in round() is banker's rounding: round(5.25, 1) is 5.2, not 5.3. That
    difference produced two wrong printed numbers in a sibling repo, so every number that
    reaches a table goes through this instead.
    """
    from decimal import Decimal, ROUND_HALF_UP
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    quant = Decimal(1).scaleb(-decimals)
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


def assert_unique_keys(df, key_cols, label=''):
    """Fail loudly if key_cols does not uniquely identify a row.

    A merge on a non-unique key silently produces the cross product: k rows against k rows
    becomes k^2 pairs. That is invisible in the output, which just looks like a larger and
    more reassuring n, and it corrupted a headline table in a sibling repo. Every merge and
    every paired subtraction in this repo calls this first.
    """
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
