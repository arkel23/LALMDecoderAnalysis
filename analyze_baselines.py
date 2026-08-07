"""Serial 10: off-the-shelf LALM baselines -- is training a connector worth doing at all?

Three serials, three measurements, and only one pair is comparable:

  serial 0   training runs. eval/cer on the model-SELECTION curve, so not held out.
  serial 10  baselines, on FLEURS/WorldSpeech test.
  serial 11  the trained checkpoints over the SAME configs as serial 10.

So 11 minus 10 is the like-for-like "what did training buy" contrast, and this script computes
it as soon as 11 exists. Serial 0 still does not join that comparison: it is a different split
and a selection curve, not a held-out number.

Both sweeps now pass `--eval_metrics wer_all cer`, so WER and CER are available on both sides --
the upstream default is `['wer_all']` alone, which is why earlier baseline runs logged no CER.
`--metric` chooses which drives the contrast.

Handles partial data: serial 10 fills in incrementally, so every cell is reported as
present/absent rather than assumed.

Usage:
    python analyze_baselines.py --input_file data/raw_serials/raw_serial_10.csv \
        --output_file results_all/acc/t7_baselines.csv
"""
import os
import argparse

import numpy as np
import pandas as pd

from missing_runs import REGISTRY
from utils import (LANGUAGE_DIC, RESOURCE_TIER, get_eval_domain, to_study_cell,
                   in_domain_role, IN_DOMAIN_PRIMARY, parent_model_id, checkpoint_fields,
                   assert_unique_keys)

FLOAT_FORMAT = '%.6f'

# wer is primary; mer/wil/wip disambiguate failure modes -- high wer with low mer means
# insertions/deletions rather than substitutions, which is what prompt-ignoring produces. cer is
# carried because the training runs report CER, so it is what makes serial 11 comparable to them.
WER_FAMILY = ('wer', 'mer', 'wil', 'wip', 'cer')

# A baseline that wins on WER while being 7x larger is not the same result as one at parity.
COST_COLS = ('rtfx', 'no_params', 'n_params', 'total_MB', 'bpw', 'max_memory')


def load_serial(path, label):
    if not os.path.exists(path):
        print(f'[SKIP] {label}: {path} not present')
        return None
    df = pd.read_csv(path)
    # A trained checkpoint's model_id names the checkpoint, so serial 11 joins to nothing until
    # it is reduced to its parent. Parent ids pass through unchanged, so this is safe for both.
    if 'model_id' in df.columns:
        df['checkpoint_id'] = df['model_id']
        df['model_id'] = df['model_id'].map(parent_model_id)
        fields = df['checkpoint_id'].map(checkpoint_fields)
        df['checkpoint_lang'] = [f[0] for f in fields]
        df['checkpoint_step'] = [f[1] for f in fields]
    print(f'{label}: {len(df)} rows from {path}')
    return df


def build_table(df):
    keep = ['serial', 'model_id', 'checkpoint_id', 'checkpoint_lang', 'checkpoint_step',
            'dataset', 'dataset_path', 'split', 'state',
            'force_asr_language', 'num_samples', 'audio_length_s_mean']
    keep += [c for c in WER_FAMILY + COST_COLS if c in df.columns]
    out = df[[c for c in keep if c in df.columns]].copy()

    # WorldSpeech configs are named after the trained variety (fr_ca), FLEURS after the
    # evaluated one (fr_fr). Normalise so both domains of a language group together.
    out['study_cell'] = out['dataset'].map(to_study_cell)
    out['language_name'] = out['study_cell'].map(LANGUAGE_DIC)
    out['resource_tier'] = out['study_cell'].map(RESOURCE_TIER)
    out['model_short'] = out['model_id'].astype(str).str.split('/').str[-1]

    # A property of the EVAL SET, not of the language.
    out['eval_domain'] = np.where(
        out['dataset_path'].astype(str).str.contains('WorldSpeech'),
        'in_domain', 'cross_domain')

    # All 33 variants normalise to the same study cell, so a cell can hold 9 in-domain rows.
    # Exactly one is its in-domain point; averaging would mix it with accent transfer.
    out['in_domain_role'] = [in_domain_role(c, d, e) for c, d, e
                             in zip(out['study_cell'], out['dataset'], out['eval_domain'])]

    # One FINISHED row per (model, language, eval set). A failed attempt sitting beside its
    # retry is a re-run, not a duplicate eval, so only finished rows carry the constraint.
    key = ['model_id', 'dataset', 'dataset_path', 'split']
    assert_unique_keys(out[out['state'] == 'finished'], key, label='t7_baselines')
    retried = out[out.duplicated(key, keep=False)]
    if len(retried):
        print(f'{retried[key].drop_duplicates().shape[0]} eval config(s) have a re-run; '
              f'states: {sorted(retried["state"].unique())}')
    return out.sort_values(['dataset', 'model_short'])


def evaluable_cells(registry_path=REGISTRY):
    """(study_cell, eval_domain) pairs the registry actually has a config for.

    FLEURS has no Seychellois Creole, so crs_sc/cross_domain can never be run. Reporting it as
    'missing' reads as 'not yet', which is a different thing from 'does not exist'.
    """
    if not os.path.exists(registry_path):
        return None
    reg = pd.read_csv(registry_path)
    reg = reg[reg['use_in_sweep']]
    domain = np.where(reg['source'].astype(str).str.contains('fleurs'),
                      'cross_domain', 'in_domain')
    return set(zip(reg['study_cell'].map(to_study_cell), domain))


def coverage(out):
    """Split what is absent into three kinds, because they mean different things.

    not_applicable  the registry has no config -- it can never be run
    failed          a run exists and failed. On crs_sc that IS the finding: the model does not
                    support the language, so it is not a gap to be filled later
    pending         no run yet, the only kind that fills in on its own
    """
    fin = out[out['state'] == 'finished']
    models = sorted(out['model_short'].dropna().unique())
    langs = sorted(out['study_cell'].dropna().unique())
    grid = pd.crosstab([fin['study_cell'], fin['eval_domain']], fin['model_short'])

    evaluable = evaluable_cells()
    absent = {'not_applicable': [], 'failed': [], 'pending': []}
    for lang in langs:
        for dom in ('cross_domain', 'in_domain'):
            for model in models:
                cell = ((out['study_cell'] == lang) & (out['eval_domain'] == dom)
                        & (out['model_short'] == model))
                if (cell & (out['state'] == 'finished')).any():
                    continue
                if evaluable is not None and (lang, dom) not in evaluable:
                    kind = 'not_applicable'
                elif (cell & (out['state'] == 'failed')).any():
                    kind = 'failed'
                else:
                    kind = 'pending'
                absent[kind].append((lang, dom, model))
    return models, langs, grid, absent


def compare_with_trained(base, trained, metric='cer'):
    """serial 11 minus serial 10, per (language, eval set). Same metric, same split.

    Returns None until the trained sweep exists. The merge asserts key uniqueness on both
    sides first: a duplicated key here would produce a cross product that reads as a larger,
    more reassuring sample.
    """
    if trained is None or base is None or metric not in trained.columns:
        return None
    # Both sides go through build_table, so model_short / study_cell / in_domain_role exist on
    # each. Passing the raw frame here is what made this path fail the first time it ran.
    trained = build_table(trained)

    key = ['dataset', 'dataset_path', 'split']
    b = base[base['state'] == 'finished'].copy()
    t = trained[trained['state'] == 'finished'].copy()

    # The honest comparator is the strongest downloadable model, not the mean of them.
    b_best = (b.sort_values(metric).groupby(key, as_index=False)
              .first()[key + [metric, 'model_short']]
              .rename(columns={metric: f'baseline_{metric}',
                               'model_short': 'baseline_model'}))
    # Best trained model per eval cell.
    t_best = (t.sort_values(metric).groupby(key, as_index=False)
              .first()[key + [metric, 'model_short']]
              .rename(columns={metric: f'trained_{metric}',
                               'model_short': 'trained_model'}))

    assert_unique_keys(b_best, key, label='baseline best-per-cell')
    assert_unique_keys(t_best, key, label='trained best-per-cell')

    merged = b_best.merge(t_best, on=key, how='inner', validate='one_to_one')
    merged[f'delta_{metric}'] = merged[f'trained_{metric}'] - merged[f'baseline_{metric}']
    merged['training_helps'] = merged[f'delta_{metric}'] < 0

    # Per eval config, so nothing is averaged -- but the reader needs to know which row is
    # the cell's in-domain point.
    merged['study_cell'] = merged['dataset'].map(to_study_cell)
    merged['eval_domain'] = np.where(
        merged['dataset_path'].astype(str).str.contains('WorldSpeech'),
        'in_domain', 'cross_domain')
    merged['in_domain_role'] = [in_domain_role(c, d, e) for c, d, e
                                in zip(merged['study_cell'], merged['dataset'],
                                       merged['eval_domain'])]
    return merged.sort_values(['study_cell', 'eval_domain', 'dataset'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_file', type=str,
                   default=os.path.join('data', 'raw_serials', 'raw_serial_10.csv'))
    p.add_argument('--trained_file', type=str,
                   default=os.path.join('data', 'raw_serials', 'raw_serial_11.csv'),
                   help='Serial 11: the trained checkpoints swept over the same FLEURS test '
                        'configs. Optional; the contrast is skipped until it exists.')
    p.add_argument('--output_file', type=str,
                   default=os.path.join('results_all', 'acc', 't7_baselines.csv'))
    p.add_argument('--contrast_file', type=str,
                   default=os.path.join('results_all', 'acc', 't7_training_vs_baseline.csv'))
    # CER is the study's primary error rate everywhere else (serial 0 selects on
    # eval/cer), so the baseline contrast uses it too. WER stays in the CSV.
    p.add_argument('--metric', type=str, default='cer')
    args = p.parse_args()

    base_raw = load_serial(args.input_file, 'serial 10 (baselines)')
    if base_raw is None:
        print('Nothing to do until serial 10 has been downloaded.')
        return 0

    out = build_table(base_raw)
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    out.to_csv(args.output_file, index=False, float_format=FLOAT_FORMAT)
    print(f'Wrote {len(out)} rows to {args.output_file}\n')

    models, langs, grid, absent = coverage(out)
    print(f'models: {", ".join(models)}')
    print(f'languages: {", ".join(langs)}')
    print('\nfinished runs per (language, model):')
    print(grid.to_string() if len(grid) else '  (none finished yet)')

    EXPLAIN = {
        'not_applicable': 'no config in the registry -- CANNOT be run, not a gap',
        'failed': 'run exists and failed -- on crs_sc that is the result, not a gap',
        'pending': 'no run yet -- this is the only kind that fills in',
    }
    for kind in ('not_applicable', 'failed', 'pending'):
        cells = absent[kind]
        if not cells:
            continue
        print(f'\n{len(cells)} cell(s) {kind}: {EXPLAIN[kind]}')
        for dom in ('cross_domain', 'in_domain'):
            names = [f'{l}/{m}' for l, d, m in cells if d == dom]
            if names:
                print(f'  {dom}: {", ".join(names[:10])}'
                      f'{" ..." if len(names) > 10 else ""}')
    if absent['pending']:
        print('\nAggregates below cover ONLY what has finished.')

    fin = out[out['state'] == 'finished']
    if len(fin):
        show = ['study_cell', 'eval_domain', 'model_short', 'resource_tier'] + \
               [c for c in WER_FAMILY if c in fin.columns] + \
               [c for c in ('rtfx', 'no_params', 'num_samples') if c in fin.columns]
        print('\nfinished baseline results:')
        print(fin.sort_values(['eval_domain', args.metric])[show].round(3).to_string(index=False))

        # Same model, same language, two domains: the cleanest read on how much harder the
        # in-domain corpus is than the read-speech benchmark.
        primary = fin[fin['in_domain_role'] != 'accent_transfer']
        assert_unique_keys(primary[primary['eval_domain'] == 'in_domain'],
                           ['model_id', 'study_cell'], label='in-domain primary rows')
        both = (primary.pivot_table(index=['study_cell', 'model_short'], columns='eval_domain',
                                    values=args.metric)
                .dropna())
        if len(both):
            both['in_over_cross'] = both['in_domain'] / both['cross_domain']
            print('\nlanguages evaluated on BOTH domains (WER):')
            print(both.round(2).to_string())

    trained_raw = load_serial(args.trained_file, 'serial 11 (trained)')
    contrast = compare_with_trained(out, trained_raw, args.metric)
    if contrast is None:
        print(f'\n[PENDING] the training-vs-baseline contrast needs serial 11 '
              f'({args.trained_file}). Run eval_lalm_decoder_txf.sh, download it, and re-run.')
    else:
        contrast.to_csv(args.contrast_file, index=False, float_format=FLOAT_FORMAT)
        print(f'\nWrote {len(contrast)} rows to {args.contrast_file}')
        print(contrast.round(3).to_string(index=False))
        n_help = int(contrast['training_helps'].sum())
        print(f'\nTraining beats the best downloadable baseline in '
              f'{n_help}/{len(contrast)} evaluated cells.')

    return 0


if __name__ == '__main__':
    main()
