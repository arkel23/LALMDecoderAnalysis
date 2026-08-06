"""Per-run curve statistics from the 101 logged evaluations, rather than the single endpoint.

Run-to-run noise exceeds the effect being looked for, so a table of final CERs mostly reports
noise. best_cer is primary (the last checkpoint is not the best -- mean gap ~3.5 CER);
final_minus_best measures overfitting; audio_h_to_*x_best is the sample-efficiency statistic;
late_sd is the within-run noise floor.

audio_hours is audio PROCESSED, counting repeats. The wandb train/epoch counter is unreliable
under streaming, so epochs_logged is carried only to be flagged, never used as a divisor.

Usage:
    python analyze_sample_efficiency.py \
        --input_file data/raw_serials/history_serial_{0,2}.csv \
        --output_file results_all/acc/t1_sample_efficiency.csv

Spans serials 0 (the grid) and 2 (the control arms), so the crs_sc cell keeps all six models.
`serial` is a column; cross-language aggregates must restrict to the grid.
"""
import os
import argparse
import numpy as np
import pandas as pd

from utils import (add_language_columns, assert_unique_keys, is_excluded_from_aggregate,
                   MODEL_SHORT, CURVE_SERIALS, GRID_SERIAL)


# Far enough above the optimum to be reached before the curve flattens, so they measure how
# fast a run got good rather than where it stopped.
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
            'seed': first.get('seed'),
            # Earliest logged wall-clock for the run. Only used to break ties when one cell
            # holds two runs, so the choice is deterministic rather than groupby order.
            'run_start': g['_timestamp'].min(),
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

    out = mark_canonical(out)

    # Keyed on serial too: t1 spans serials 0 and 2, and the same (model, language) can appear
    # in both only if a control arm leaked into the grid. Asserted on the canonical subset
    # because during a re-run window a cell legitimately holds two runs in one serial.
    assert_unique_keys(out[out['is_canonical']], ['model_id', 'dataset', 'serial'],
                       label='t1_sample_efficiency (canonical rows)')

    return out.sort_values(['dataset', 'model_short', 'run_start'])


def mark_canonical(out):
    """Pick one run per (model_id, dataset), and say so loudly when there was a choice.

    A cell holds two runs in the same serial only transiently: a re-run has started but the
    original has not yet been migrated to serial 1. The pipeline must not fall over during that
    window, and it must not silently pick a different run than the one every existing number
    came from.

    So the rule is stability-first: prefer a `finished` run over an unfinished one, and among
    equals prefer the EARLIEST. A half-trained re-run can never displace the completed run it
    is meant to replace, and once it finishes the swap is a deliberate act -- migrating the
    original to serial 1 -- rather than something that happens on the next `bash plotter.sh`.

    Every duplicate is printed. A silent de-duplication here would be worse than the crash it
    replaces, because the dropped run leaves no trace in the output.
    """
    out = out.copy()
    order = out.assign(_unfinished=(out['state'] != 'finished').astype(int))
    order = order.sort_values(['_unfinished', 'run_start'], kind='mergesort')
    keep = order.groupby(['model_id', 'dataset', 'serial'], sort=False).head(1)['run_id']

    out['is_canonical'] = out['run_id'].isin(set(keep))
    out['n_runs_in_cell'] = (out.groupby(['model_id', 'dataset', 'serial'])['run_id']
                             .transform('size'))

    dupes = out[out['n_runs_in_cell'] > 1]
    if not dupes.empty:
        cells = dupes[['model_id', 'dataset', 'serial']].drop_duplicates()
        print(f'\n[DUPLICATE RUNS] {len(cells)} cell(s) hold more than one run in this serial. '
              f'A serial migration is pending -- move the superseded run to serial 1 with '
              f'rename_wandb_serial.py once its replacement has finished.')
        cols = ['dataset', 'model_short', 'run_id', 'state', 'seed', 'best_cer', 'is_canonical']
        shown = dupes.sort_values(['dataset', 'model_short', 'run_start'])
        print(shown[[c for c in cols if c in shown.columns]].to_string(index=False))
        print()
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_file', nargs='+', type=str,
                   default=[os.path.join('data', 'raw_serials', f'history_serial_{s}.csv')
                            for s in CURVE_SERIALS],
                   help='One or more history CSVs; concatenated, with serial kept as a column.')
    p.add_argument('--output_file', type=str,
                   default=os.path.join('results_all', 'acc', 't1_sample_efficiency.csv'))
    p.add_argument('--keep_finished_only', action='store_true',
                   help='Drop runs that are still running (their curves are truncated).')
    return p.parse_args()


def main():
    args = parse_args()
    paths = [f for f in args.input_file if os.path.exists(f)]
    missing = [f for f in args.input_file if not os.path.exists(f)]
    for f in missing:
        print(f'[SKIP] {f} not present')
    df = pd.concat([pd.read_csv(f) for f in paths], ignore_index=True)
    print(f'Read {len(df)} step-rows from {len(paths)} file(s): {", ".join(paths)}')

    if args.keep_finished_only:
        before = df['run_id'].nunique()
        df = df[df['state'] == 'finished']
        print(f'Kept {df["run_id"].nunique()}/{before} finished runs')

    out = build_table(df)

    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    out.to_csv(args.output_file, index=False, float_format=FLOAT_FORMAT)
    print(f'Wrote {len(out)} rows to {args.output_file}')

    # Summarised over the GRID only. t1 also carries serial 2's control arms, and pooling them
    # into a cross-language statistic is the mistake this serial split exists to prevent.
    fin = out[~out['excluded_from_aggregate'] & (out['state'] == 'finished')
              & (out['serial'] == GRID_SERIAL) & out['is_canonical']]
    print(f'\ngrid (serial {GRID_SERIAL}) statistics over {len(fin)} runs:')
    print(f'median within-run late_sd     : {fin["late_sd"].median():.3f} CER')
    print(f'mean final_minus_best         : {fin["final_minus_best"].mean():.3f} CER')
    print(f'max  final_minus_best         : {fin["final_minus_best"].max():.3f} CER')
    return 0


if __name__ == '__main__':
    main()
