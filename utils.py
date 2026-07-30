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
    # 600: 'Connector-only SFT',   # --freeze_encoder --freeze_decoder (SLAM-style)
    # 601: 'Decoder LoRA',
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
METHODS_DIC = {
    'q2a_whisper_small_tiny_aya_base': 'TinyAya-base',
    'q2a_whisper_small_tiny_aya_global': 'TinyAya-global',
    'q2a_whisper_small_tiny_aya_earth': 'TinyAya-earth',
    'q2a_whisper_small_tiny_aya_fire': 'TinyAya-fire',
    'q2a_whisper_small_tiny_aya_water': 'TinyAya-water',
}

# Grouping for figures that aggregate variants (e.g. regional vs non-regional).
MODEL_FAMILY = {
    'q2a_whisper_small_tiny_aya_base': 'Non-regional',
    'q2a_whisper_small_tiny_aya_global': 'Non-regional',
    'q2a_whisper_small_tiny_aya_earth': 'Regional',
    'q2a_whisper_small_tiny_aya_fire': 'Regional',
    'q2a_whisper_small_tiny_aya_water': 'Regional',
}

# Models logged but excluded from aggregates, with the reason. Keep the raw data; exclude
# at analysis time only, and state the reason in the paper.
EXCLUDED_MODELS_AGGREGATE = []

# --- Datasets and metrics -----------------------------------------------------------
DATASETS_DIC = {}

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
