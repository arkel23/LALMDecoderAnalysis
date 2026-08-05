"""Loss diagnostics: separating overfitting from domain and accent shift, which CER cannot.

  generalisation_gap = eval_loss_final - train_loss_final
      Inflated by BOTH overfitting and distribution shift, so not an overfitting measure alone.

  eval_loss_final_minus_best = eval_loss_final - eval_loss_best
      Eval loss rising after its own minimum: overfitting proper, unaffected by domain or accent
      shift. A large gap with ~zero rise means shifted, not overfit.

Read against EVAL_DOMAIN, ACCENT_MATCH and RESOURCE_TIER.

Eval loss is comparable across models WITHIN a language, never across languages -- tokenisation
and sequence length differ, so cross-entropies are not commensurable. Every aggregate is a
within-language contrast or grouped by language.

Usage:
    python analyze_loss_metrics.py --input_file data/raw_serials/history_serial_0.csv \
        --output_file results_all/acc/t6_loss_metrics.csv
"""
import os
import argparse

import numpy as np
import pandas as pd

from utils import (LANGUAGE_DIC, LANGUAGE_REGION, MODEL_SHORT, RESOURCE_TIER, TIER_ORDER,
                   get_eval_domain, get_accent_match, assert_unique_keys)

LATE_FRACTION = 0.3
FLOAT_FORMAT = '%.6f'


def curve_losses(g):
    """Loss statistics for one run, from its interleaved train- and eval-log rows."""
    g = g.sort_values('audio_hours')
    ev = g[g['eval/loss'].notna()]
    tr = g[g['train/loss'].notna()]
    if ev.empty or tr.empty:
        return None

    eval_loss = ev['eval/loss'].to_numpy(dtype=float)
    hours = ev['audio_hours'].to_numpy(dtype=float)
    best_i = int(np.nanargmin(eval_loss))

    train_final = float(tr['train/loss'].iloc[-1])
    eval_final = float(eval_loss[-1])
    eval_best = float(eval_loss[best_i])

    late = eval_loss[hours >= hours[-1] * (1.0 - LATE_FRACTION)]

    return {
        'train_loss_final': train_final,
        'train_loss_best': float(tr['train/loss'].min()),
        'eval_loss_final': eval_final,
        'eval_loss_best': eval_best,
        'audio_h_to_best_eval_loss': float(hours[best_i]),
        'frac_of_run_to_best_eval_loss': float(hours[best_i] / hours[-1]) if hours[-1] else np.nan,
        # Inflated by overfitting AND by distribution shift.
        'generalisation_gap': eval_final - train_final,
        # Overfitting proper: eval loss rising after its own minimum.
        'eval_loss_final_minus_best': eval_final - eval_best,
        'eval_loss_late_sd': float(np.std(late, ddof=1)) if late.size > 1 else np.nan,
        'n_eval_points': int(len(ev)),
    }


def build_table(df):
    rows = []
    for run_id, g in df.groupby('run_id', sort=False):
        stats = curve_losses(g)
        if stats is None:
            continue
        first = g.iloc[0]
        lang = first.get('dataset')
        rec = {
            'run_id': run_id,
            'dataset': lang,
            'language_name': LANGUAGE_DIC.get(lang),
            'region': LANGUAGE_REGION.get(lang),
            'model_short': MODEL_SHORT.get(first.get('model_id')),
            'model_id': first.get('model_id'),
            'state': first.get('state'),
            'eval_domain': get_eval_domain(lang),
            'accent_match': get_accent_match(lang),
            'resource_tier': RESOURCE_TIER.get(lang),
            'eval_dataset_path': first.get('dataset_path'),
            'eval_split': first.get('split'),
        }
        rec.update(stats)
        rows.append(rec)

    out = pd.DataFrame(rows)
    assert_unique_keys(out, ['run_id'], label='t6_loss_metrics')
    return out


def summarise(out):
    """Group the two loss quantities by each design axis.

    Losses are not comparable across languages, so every group is reported with its language
    count and the aggregates are means over per-run values within the group -- useful for a
    contrast between axes, not as an absolute scale.
    """
    fin = out[out['state'] == 'finished']
    frames = []
    for axis in ('eval_domain', 'accent_match', 'resource_tier'):
        g = (fin.groupby(axis, as_index=False)
             .agg(n_runs=('run_id', 'count'),
                  n_languages=('dataset', 'nunique'),
                  mean_generalisation_gap=('generalisation_gap', 'mean'),
                  median_generalisation_gap=('generalisation_gap', 'median'),
                  mean_eval_loss_rise=('eval_loss_final_minus_best', 'mean'),
                  median_eval_loss_rise=('eval_loss_final_minus_best', 'median'),
                  mean_frac_to_best=('frac_of_run_to_best_eval_loss', 'mean')))
        g.insert(0, 'axis', axis)
        g = g.rename(columns={axis: 'level'})
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_file', type=str,
                   default=os.path.join('data', 'raw_serials', 'history_serial_0.csv'))
    p.add_argument('--output_file', type=str,
                   default=os.path.join('results_all', 'acc', 't6_loss_metrics.csv'))
    p.add_argument('--summary_file', type=str,
                   default=os.path.join('results_all', 'acc', 't6_loss_by_axis.csv'))
    args = p.parse_args()

    df = pd.read_csv(args.input_file)
    out = build_table(df)

    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    out.to_csv(args.output_file, index=False, float_format=FLOAT_FORMAT)
    print(f'Wrote {len(out)} rows to {args.output_file}')

    summary = summarise(out)
    summary.to_csv(args.summary_file, index=False, float_format=FLOAT_FORMAT)
    print(f'Wrote {len(summary)} rows to {args.summary_file}\n')

    print(summary[['axis', 'level', 'n_runs', 'n_languages', 'median_generalisation_gap',
                   'median_eval_loss_rise', 'mean_frac_to_best']].round(3).to_string(index=False))

    fin = out[out['state'] == 'finished']
    print('\nper-language medians (losses are NOT comparable across languages):')
    per_lang = (fin.groupby(['dataset', 'resource_tier', 'eval_domain'], as_index=False)
                .agg(gen_gap=('generalisation_gap', 'median'),
                     eval_rise=('eval_loss_final_minus_best', 'median'),
                     frac_to_best=('frac_of_run_to_best_eval_loss', 'median')))
    per_lang['tier_order'] = per_lang['resource_tier'].map(
        {t: i for i, t in enumerate(TIER_ORDER)})
    print(per_lang.sort_values('tier_order').drop(columns='tier_order')
          .round(3).to_string(index=False))
    return 0


if __name__ == '__main__':
    main()
