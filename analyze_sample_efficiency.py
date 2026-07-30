"""
Per-run curve statistics: the table that makes the decoder comparison say something the
final-CER number cannot.

Why this exists. Each run's endpoint is a single number, and the measured run-to-run noise
is larger than the effect being looked for, so a table of final CERs mostly reports noise.
Each run also logs 101 evaluations spanning ~21 to ~2700 hours of processed audio. Reducing
that curve to a handful of statistics uses all 101 points instead of 1:

  best_cer            lowest CER reached at any evaluation. The primary metric -- the last
                      checkpoint is not the best one (mean final-minus-best gap is ~3.5 CER).
  final_cer           CER at the last evaluation. Secondary, and reported as such.
  final_minus_best    how much was given back after the optimum. This IS the overfitting
                      measure, and it should be largest where a language cannot fill one
                      clean epoch.
  audio_h_to_best     processed audio at the optimum.
  audio_h_to_*x_best  processed audio needed to first come within a factor of the optimum.
                      The sample-efficiency statistic: it separates variants far better than
                      final CER because it is not a single noisy endpoint.
  late_sd             sd of CER over the last 30% of the curve. This is the within-run noise
                      floor, and it is the only error bar available without extra seeds. Any
                      claimed difference smaller than it is not a difference.

Note on the x-axis: audio_hours is audio PROCESSED (cumulative, counting repeats), not
unique audio. train/train_audio_seconds is a real measurement -- it sums decoded array
length / sampling rate per batch -- but the wandb train/epoch counter is unreliable under
streaming, so unique-hours and epoch counts are NOT derived here. epochs_logged is carried
through only so it can be flagged, never used as a divisor.

Usage:
    python analyze_sample_efficiency.py --input_file data/raw_serials/history_serial_0.csv \
        --output_file results_all/acc/t1_sample_efficiency.csv
"""
import os
import argparse
import numpy as np
import pandas as pd

from utils import (add_language_columns, assert_unique_keys, is_excluded_from_aggregate,
                   MODEL_SHORT)


# Fractions of the best CER used as sample-efficiency thresholds. 1.5x and 2.0x are far
# enough above the optimum to be reached well before the curve flattens, so they measure
# how fast a run got good rather than where it happened to stop.
THRESHOLD_FACTORS = (1.25, 1.5, 2.0)

# Fraction of the curve (by processed audio) treated as "late training" for the noise floor.
LATE_FRACTION = 0.3

FLOAT_FORMAT = '%.6f'


def curve_stats(ev):
    """ev: the eval rows of one run, sorted by audio_hours."""
    cer = ev['eval/cer'].to_numpy(dtype=float)
    hours = ev['audio_hours'].to_numpy(dtype=float)

    best_i = int(np.nanargmin(cer))
    best_cer = float(cer[best_i])
    final_cer = float(cer[-1])

    out = {
        'n_evals': int(len(ev)),
        'best_cer': best_cer,
        'final_cer': final_cer,
        'final_minus_best': final_cer - best_cer,
        'audio_h_to_best': float(hours[best_i]),
        'audio_h_total': float(hours[-1]),
        'first_cer': float(cer[0]),
    }

    for factor in THRESHOLD_FACTORS:
        target = best_cer * factor
        hit = np.flatnonzero(cer <= target)
        key = f'audio_h_to_{factor:g}x_best'
        out[key] = float(hours[hit[0]]) if hit.size else np.nan

    # Noise floor: spread of CER once the run has stopped improving quickly.
    cutoff = hours[-1] * (1.0 - LATE_FRACTION)
    late = cer[hours >= cutoff]
    out['late_sd'] = float(np.std(late, ddof=1)) if late.size > 1 else np.nan
    out['late_mean'] = float(np.mean(late)) if late.size else np.nan
    out['n_late'] = int(late.size)

    return out


def build_table(df):
    rows = []
    for run_id, g in df.groupby('run_id', sort=False):
        ev = g[g['eval/cer'].notna()].sort_values('audio_hours')
        if ev.empty:
            continue
        first = g.iloc[0]
        rec = {
            'run_id': run_id,
            'serial': first.get('serial'),
            'model_id': first.get('model_id'),
            'model_short': MODEL_SHORT.get(first.get('model_id')),
            'dataset': first.get('dataset'),
            'dataset_path': first.get('dataset_path'),
            'split': first.get('split'),
            'state': first.get('state'),
            'effective_batch': first.get('effective_batch'),
            # Carried, flagged, and never used as a divisor -- see the module docstring.
            'epochs_logged_unreliable': ev['train/epoch'].max(),
        }
        rec.update(curve_stats(ev))
        rows.append(rec)

    out = pd.DataFrame(rows)
    out = add_language_columns(out, lang_col='dataset')
    out['excluded_from_aggregate'] = [
        is_excluded_from_aggregate(m, d) for m, d in zip(out['model_id'], out['dataset'])
    ]

    # One row per (model, language) is the contract every downstream merge relies on.
    assert_unique_keys(out, ['model_id', 'dataset'], label='t1_sample_efficiency')

    return out.sort_values(['dataset', 'model_short'])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_file', type=str,
                   default=os.path.join('data', 'raw_serials', 'history_serial_0.csv'))
    p.add_argument('--output_file', type=str,
                   default=os.path.join('results_all', 'acc', 't1_sample_efficiency.csv'))
    p.add_argument('--keep_finished_only', action='store_true',
                   help='Drop runs that are still running (their curves are truncated).')
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input_file)

    if args.keep_finished_only:
        before = df['run_id'].nunique()
        df = df[df['state'] == 'finished']
        print(f'Kept {df["run_id"].nunique()}/{before} finished runs')

    out = build_table(df)

    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    out.to_csv(args.output_file, index=False, float_format=FLOAT_FORMAT)
    print(f'Wrote {len(out)} rows to {args.output_file}')

    fin = out[~out['excluded_from_aggregate'] & (out['state'] == 'finished')]
    print(f'\nmedian within-run late_sd     : {fin["late_sd"].median():.3f} CER')
    print(f'mean final_minus_best         : {fin["final_minus_best"].mean():.3f} CER')
    print(f'max  final_minus_best         : {fin["final_minus_best"].max():.3f} CER')
    return 0


if __name__ == '__main__':
    main()
