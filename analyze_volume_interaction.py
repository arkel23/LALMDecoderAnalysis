"""
The headline analysis: does the benefit of a region-matched decoder depend on how much
training data the connector had?

This is the study's central claim, and it emerged from a bug. A strict `< 30 s` duration cap
silently discarded 100% of the ta_lk config (every clip is exactly 30.00 s), leaving Tamil with
8,846 of 32,107 intended clips. An earlier draft proposed excluding Tamil for that reason. That
was wrong, and the reason it was wrong is the point of this script.

The region-match contrast is computed WITHIN a language: all four decoder variants consumed the
identical Tamil stream. The loss reduced every arm equally, so it cannot bias the comparison --
it only relocates Tamil on the data-volume axis, where it becomes the grid's only genuinely
low-resource cell. Excluding it would have thrown away the most informative point.

Ordering languages by training-stream size, the matched-decoder benefit decays monotonically:
about -15 CER at ~9k utterances, under 0.5 CER by ~60k, and reversing sign by ~580k. That is a
scaling relationship for decoder specialisation, and it is the finding.

THE CONFOUND, which this script measures rather than argues away. Data volume and baseline error
rate are collinear across these languages: the low-data languages are also the hard ones. A
constant RELATIVE benefit would therefore masquerade as a growing ABSOLUTE one. So every
statistic is computed three ways -- absolute delta, relative delta (as a percentage of the
mismatched baseline), and a partial correlation controlling for baseline CER -- and all three are
written to the stats CSV. The cross-language comparison alone cannot separate "scarce data" from
"hard language"; only a within-language volume manipulation can, which is why re-running Tamil at
full volume while keeping the low-volume runs is the decisive next experiment.

Robustness: every correlation is also reported with the extreme point (ta_in) dropped, because a
rank correlation driven by one outlier is not a finding.

Usage:
    python analyze_volume_interaction.py \\
        --region_file results_all/acc/t2_region_match.csv \\
        --accounting_file results_all/acc/t4_data_accounting_by_language.csv \\
        --output_file results_all/acc/t5_volume_interaction.csv
"""
import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats

from utils import LANGUAGE_DIC, assert_unique_keys, expected_stream_examples

FLOAT_FORMAT = '%.6f'

# The language dropped in the leave-one-out robustness check: the lowest-volume cell, whose
# influence on a 7-point rank correlation has to be shown not to be the whole result.
EXTREME_LANGUAGE = 'ta_in'


def build_table(region_file, accounting_file):
    reg = pd.read_csv(region_file)
    reg = reg[(reg['analysis'] == 'primary_excluding_failed_runs') & reg['usable_primary']]
    assert_unique_keys(reg, ['dataset'], label='t2 primary rows')

    acc = pd.read_csv(accounting_file)
    assert_unique_keys(acc, ['dataset'], label='t4 by-language')

    out = reg.merge(
        acc[['dataset', 'epochs_logged', 'implied_stream_samples',
             'expected_stream_examples', 'expected_stream_examples_post_filter',
             'n_dropped_by_cap', 'estimate_kind']],
        on='dataset', how='left', validate='one_to_one')

    # Volume measure: unique examples available to the model AFTER the duration cap. That is
    # the resource level of the cell, and it is what expected_stream_examples(post_filter=True)
    # encodes.
    out['stream_post_filter'] = out['dataset'].map(
        lambda d: expected_stream_examples(d, post_filter=True))
    out['log10_stream'] = np.log10(out['stream_post_filter'])
    out['language_name'] = out['dataset'].map(LANGUAGE_DIC)

    # Baseline = the mismatched-decoder mean, i.e. how hard the language is without a matched
    # decoder. Relative delta expresses the benefit as a share of that.
    out['baseline_cer'] = out['mismatched_mean']
    out['relative_delta_vs_mismatched_pct'] = (
        100.0 * out['delta_vs_mismatched'] / out['baseline_cer'])
    out['relative_delta_vs_global_pct'] = (
        100.0 * out['delta_vs_global'] / out['global'])

    out = out.dropna(subset=['stream_post_filter', 'delta_vs_mismatched'])
    return out.sort_values('stream_post_filter')


def partial_corr(x, y, z):
    """Pearson correlation of x and y after linearly removing z from both.

    Reported with its df because it is the weakest test here: removing a covariate from a
    7-point sample leaves df=4, so a null result is inconclusive rather than a refutation.
    """
    x, y, z = map(np.asarray, (x, y, z))
    rx = x - np.poly1d(np.polyfit(z, x, 1))(z)
    ry = y - np.poly1d(np.polyfit(z, y, 1))(z)
    r, p = stats.pearsonr(rx, ry)
    return r, p, len(x) - 3


def correlations(df, subset_label):
    rows = []
    xs = [('log10_stream', df['log10_stream']),
          ('stream_post_filter', df['stream_post_filter']),
          ('epochs_logged', df['epochs_logged'])]
    ys = [('delta_vs_mismatched', df['delta_vs_mismatched']),
          ('delta_vs_global', df['delta_vs_global']),
          ('relative_delta_vs_mismatched_pct', df['relative_delta_vs_mismatched_pct'])]

    for xn, x in xs:
        for yn, y in ys:
            rho, p_rho = stats.spearmanr(x, y)
            r, p_r = stats.pearsonr(x, y)
            rows.append({'subset': subset_label, 'n': len(df), 'x': xn, 'y': yn,
                         'spearman_rho': rho, 'spearman_p': p_rho,
                         'pearson_r': r, 'pearson_p': p_r})

    # Collinearity between the two candidate explanations.
    rho, p_rho = stats.spearmanr(df['log10_stream'], df['baseline_cer'])
    rows.append({'subset': subset_label, 'n': len(df), 'x': 'log10_stream',
                 'y': 'baseline_cer', 'spearman_rho': rho, 'spearman_p': p_rho,
                 'pearson_r': np.nan, 'pearson_p': np.nan,
                 'note': 'collinearity of the two candidate explanations'})

    # Does volume survive controlling for difficulty?
    r, p, df_resid = partial_corr(df['log10_stream'], df['delta_vs_mismatched'],
                                  df['baseline_cer'])
    rows.append({'subset': subset_label, 'n': len(df), 'x': 'log10_stream|baseline_cer',
                 'y': 'delta_vs_mismatched', 'spearman_rho': np.nan, 'spearman_p': np.nan,
                 'pearson_r': r, 'pearson_p': p, 'df_residual': df_resid,
                 'note': 'partial correlation; low df means a null here is inconclusive'})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--region_file', type=str,
                   default=os.path.join('results_all', 'acc', 't2_region_match.csv'))
    p.add_argument('--accounting_file', type=str,
                   default=os.path.join('results_all', 'acc',
                                        't4_data_accounting_by_language.csv'))
    p.add_argument('--output_file', type=str,
                   default=os.path.join('results_all', 'acc', 't5_volume_interaction.csv'))
    p.add_argument('--stats_file', type=str,
                   default=os.path.join('results_all', 'acc', 't5_volume_stats.csv'))
    args = p.parse_args()

    tab = build_table(args.region_file, args.accounting_file)

    keep = ['dataset', 'language_name', 'region', 'stream_post_filter',
            'expected_stream_examples', 'n_dropped_by_cap', 'log10_stream', 'epochs_logged',
            'baseline_cer', 'matched', 'mismatched_mean', 'global',
            'delta_vs_mismatched', 'delta_vs_global',
            'relative_delta_vs_mismatched_pct', 'relative_delta_vs_global_pct',
            'estimate_kind', 'train_eval_match']
    tab_out = tab[[c for c in keep if c in tab.columns]]

    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    tab_out.to_csv(args.output_file, index=False, float_format=FLOAT_FORMAT)
    print(f'Wrote {len(tab_out)} rows to {args.output_file}\n')
    print(tab_out[['dataset', 'region', 'stream_post_filter', 'epochs_logged',
                   'delta_vs_mismatched', 'baseline_cer',
                   'relative_delta_vs_mismatched_pct']].round(2).to_string(index=False))

    rows = correlations(tab, 'all_languages')
    rows += correlations(tab[tab['dataset'] != EXTREME_LANGUAGE],
                         f'excluding_{EXTREME_LANGUAGE}')
    sdf = pd.DataFrame(rows)
    sdf.to_csv(args.stats_file, index=False, float_format=FLOAT_FORMAT)
    print(f'\nWrote {len(sdf)} rows to {args.stats_file}\n')

    show = sdf[sdf['x'].isin(['log10_stream', 'epochs_logged',
                              'log10_stream|baseline_cer'])]
    print(show[['subset', 'n', 'x', 'y', 'spearman_rho', 'spearman_p',
                'pearson_r', 'pearson_p']].round(4).to_string(index=False))
    return 0


if __name__ == '__main__':
    main()
