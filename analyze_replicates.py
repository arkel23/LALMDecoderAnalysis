"""Between-run variance, measured from replicate pairs rather than inferred from `late_sd`.

`late_sd` measures wobble along ONE trajectory; what matters when comparing two runs is how far
apart two INDEPENDENT runs of the same cell land. The one pair that exists differs by 5.01 CER
on best (en_us/water, both seed 42), against a grid-wide median late_sd of 1.06 and a
region-match effect of -0.61 -- so the quoted noise floor is roughly 5x too small.

Serial 1 holds the earlier run of a twice-run cell, so pairing is a join on (model_id, dataset).

A same-seed pair measures NONDETERMINISM, not seed sensitivity, which is strictly larger --
so it is a LOWER BOUND. `seed_status` splits pairs three ways (same_seed / seed_varies /
unrecorded) so the two are never pooled and the sentinel is never mistaken for a real seed.

Usage:
    python analyze_replicates.py --serial0_file data/raw_serials/history_serial_0.csv \
        --serial1_file data/raw_serials/history_serial_1.csv
"""
import os
import argparse

import numpy as np
import pandas as pd

from utils import LANGUAGE_DIC, MODEL_SHORT, RESOURCE_TIER, assert_unique_keys
from download_wandb_history import UNRECORDED_SEED

FLOAT_FORMAT = '%.6f'


def curve_summary(df, serial_label):
    """One row per run: best and final eval CER, plus the within-run late sd for contrast."""
    rows = []
    for run_id, g in df.groupby('run_id', sort=False):
        ev = g[g['eval/cer'].notna()].sort_values('audio_hours')
        if ev.empty:
            continue
        first = g.iloc[0]
        cer = ev['eval/cer'].to_numpy(dtype=float)
        hours = ev['audio_hours'].to_numpy(dtype=float)
        late = cer[hours >= hours[-1] * 0.7]
        rows.append({
            'serial_label': serial_label,
            'run_id': run_id,
            'model_id': first.get('model_id'),
            'dataset': first.get('dataset'),
            'state': first.get('state'),
            'seed': first.get('seed') if 'seed' in g.columns else np.nan,
            'best_cer': float(np.nanmin(cer)),
            'final_cer': float(cer[-1]),
            'late_sd': float(np.std(late, ddof=1)) if late.size > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def build_pairs(s0, s1):
    a = s0[s0['state'] == 'finished']
    b = s1[s1['state'] == 'finished']
    if a.empty or b.empty:
        return pd.DataFrame()

    assert_unique_keys(a, ['model_id', 'dataset'], label='serial 0 replicate key')
    assert_unique_keys(b, ['model_id', 'dataset'], label='serial 1 replicate key')

    pairs = b.merge(a, on=['model_id', 'dataset'], suffixes=('_first', '_rerun'),
                    how='inner', validate='one_to_one')
    if pairs.empty:
        return pairs

    pairs['language_name'] = pairs['dataset'].map(LANGUAGE_DIC)
    pairs['model_short'] = pairs['model_id'].map(MODEL_SHORT)
    pairs['resource_tier'] = pairs['dataset'].map(RESOURCE_TIER)
    pairs['delta_best_cer'] = pairs['best_cer_rerun'] - pairs['best_cer_first']
    pairs['delta_final_cer'] = pairs['final_cer_rerun'] - pairs['final_cer_first']
    pairs['abs_delta_best'] = pairs['delta_best_cer'].abs()
    # Three states, not two: a same-seed pair measures nondeterminism, a differing seed also
    # captures seed sensitivity, and the sentinel is neither. Never pool them.
    unrecorded = ((pairs['seed_first'] == UNRECORDED_SEED)
                  | (pairs['seed_rerun'] == UNRECORDED_SEED)
                  | pairs['seed_first'].isna() | pairs['seed_rerun'].isna())
    pairs['seed_varies'] = np.where(
        unrecorded, np.nan, pairs['seed_first'] != pairs['seed_rerun'])
    pairs['seed_status'] = np.where(
        unrecorded, 'unrecorded',
        np.where(pairs['seed_varies'] == 1, 'seed_varies', 'same_seed'))
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--serial0_file', type=str,
                   default=os.path.join('data', 'raw_serials', 'history_serial_0.csv'))
    p.add_argument('--serial1_file', type=str,
                   default=os.path.join('data', 'raw_serials', 'history_serial_1.csv'))
    p.add_argument('--output_file', type=str,
                   default=os.path.join('results_all', 'acc', 't9_replicates.csv'))
    p.add_argument('--stats_file', type=str,
                   default=os.path.join('results_all', 'acc', 't9_replicate_stats.csv'))
    args = p.parse_args()

    if not os.path.exists(args.serial1_file):
        print(f'[SKIP] {args.serial1_file} not present -- no replicates to pair against yet.')
        print('Serial 1 holds the earlier run of any cell run more than once. It fills as the '
              'am_et / crs_sc re-runs land and their originals are migrated.')
        return 0

    s0 = curve_summary(pd.read_csv(args.serial0_file), 'serial_0')
    s1 = curve_summary(pd.read_csv(args.serial1_file), 'serial_1')
    pairs = build_pairs(s0, s1)

    if pairs.empty:
        print('No (model_id, dataset) cell appears in both serial 0 and serial 1 yet.')
        return 0

    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    keep = ['dataset', 'language_name', 'model_short', 'resource_tier',
            'best_cer_first', 'best_cer_rerun', 'delta_best_cer',
            'final_cer_first', 'final_cer_rerun', 'delta_final_cer',
            'late_sd_first', 'late_sd_rerun', 'seed_first', 'seed_rerun', 'seed_status']
    out = pairs[[c for c in keep if c in pairs.columns]]
    out.to_csv(args.output_file, index=False, float_format=FLOAT_FORMAT)
    print(f'Wrote {len(out)} replicate pairs to {args.output_file}\n')
    print(out.round(3).to_string(index=False))

    rows = []
    for label, sub in [('all_pairs', pairs),
                       ('same_seed', pairs[pairs['seed_status'] == 'same_seed']),
                       ('seed_varies', pairs[pairs['seed_status'] == 'seed_varies']),
                       ('seed_unrecorded', pairs[pairs['seed_status'] == 'unrecorded'])]:
        if sub.empty:
            continue
        d = sub['delta_best_cer'].to_numpy(dtype=float)
        # Both members are independent draws from the same cell, hence the sqrt(2).
        between_run_sd = float(np.std(d, ddof=1) / np.sqrt(2)) if len(d) > 1 else np.nan
        rows.append({
            'subset': label,
            'n_pairs': len(sub),
            'mean_abs_delta_best': float(sub['abs_delta_best'].mean()),
            'max_abs_delta_best': float(sub['abs_delta_best'].max()),
            'sd_of_paired_difference': float(np.std(d, ddof=1)) if len(d) > 1 else np.nan,
            'between_run_sd': between_run_sd,
            'median_within_run_late_sd': float(
                pd.concat([sub['late_sd_first'], sub['late_sd_rerun']]).median()),
        })
    sdf = pd.DataFrame(rows)
    sdf.to_csv(args.stats_file, index=False, float_format=FLOAT_FORMAT)
    print(f'\nWrote {len(sdf)} rows to {args.stats_file}')
    print(sdf.round(3).to_string(index=False))

    all_row = sdf[sdf['subset'] == 'all_pairs']
    if len(all_row) and len(pairs) > 1:
        br = all_row['between_run_sd'].iloc[0]
        wr = all_row['median_within_run_late_sd'].iloc[0]
        if np.isfinite(br) and np.isfinite(wr) and wr > 0:
            print(f'\nBetween-run sd is {br / wr:.1f}x the within-run late_sd. '
                  f'Uncertainty statements based on late_sd are optimistic by that factor.')
    else:
        print(f'\nOnly {len(pairs)} pair(s) -- too few for a pooled sd. '
              f'The largest observed |delta| on best CER is '
              f'{pairs["abs_delta_best"].max():.2f} CER, against a median within-run late_sd '
              f'of {pd.concat([pairs["late_sd_first"], pairs["late_sd_rerun"]]).median():.2f}.')
    return 0


if __name__ == '__main__':
    main()
