"""Does a region-matched decoder beat a mismatched one, and beat the non-regional 'global'?

The falsifiable form of the project's claim, available for free because the languages span all
three TinyAya regions. Reports the null with a minimum detectable effect, runs a primary and a
sensitivity analysis, and flags non-clean cells (cross-dialect fr_fr, es_419) rather than
pooling them silently.

crs_sc is absent by construction -- its region is None, so it is an OOD probe. See
analyze_ood_crs.py.

Usage:
    python analyze_region_match.py --input_file results_all/acc/t1_sample_efficiency.csv \
        --output_file results_all/acc/t2_region_match.csv
"""
import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats

from utils import CORE_VARIANTS, LANGUAGE_REGION, assert_unique_keys, GRID_SERIAL


REGIONAL = ('earth', 'fire', 'water')   # 'global' is the non-regional comparator
N_BOOT = 10000
BOOT_SEED = 0
FLOAT_FORMAT = '%.6f'


def build_contrast(df, metric):
    """One row per language: matched vs mismatched-mean vs global."""
    rows = []
    for lang, g in df.groupby('dataset', sort=True):
        region = LANGUAGE_REGION.get(lang)
        if region is None:
            continue    # crs_sc: OOD probe, not a matched cell
        s = g.set_index('model_short')[metric]
        excluded = set(g.loc[g['excluded_from_aggregate'], 'model_short'])

        mismatched = [m for m in REGIONAL if m != region and m in s.index
                      and m not in excluded]
        matched_ok = region in s.index and region not in excluded

        rows.append({
            'dataset': lang,
            'region': region,
            'train_eval_match': g['train_eval_match'].iloc[0],
            'matched_variant': region,
            'matched': s.get(region, np.nan),
            'matched_excluded': region in excluded,
            'mismatched_mean': np.mean([s[m] for m in mismatched]) if mismatched else np.nan,
            'n_mismatched': len(mismatched),
            'global': s.get('global', np.nan),
            'usable_primary': bool(matched_ok and mismatched and 'global' in s.index),
        })

    out = pd.DataFrame(rows)
    out['delta_vs_mismatched'] = out['matched'] - out['mismatched_mean']
    out['delta_vs_global'] = out['matched'] - out['global']
    return out


def paired_test(deltas, label):
    """Wilcoxon signed-rank plus a bootstrap CI on the mean, over languages."""
    d = np.asarray([x for x in deltas if np.isfinite(x)], dtype=float)
    n = len(d)
    res = {'contrast': label, 'n_languages': n,
           'mean_delta': float(np.mean(d)) if n else np.nan,
           'median_delta': float(np.median(d)) if n else np.nan,
           'n_favouring_matched': int((d < 0).sum())}

    if n >= 2:
        rng = np.random.default_rng(BOOT_SEED)
        boot = rng.choice(d, size=(N_BOOT, n), replace=True).mean(axis=1)
        res['ci95_low'] = float(np.percentile(boot, 2.5))
        res['ci95_high'] = float(np.percentile(boot, 97.5))
        res['sd_delta'] = float(np.std(d, ddof=1))
        try:
            stat, p = stats.wilcoxon(d)
            res['wilcoxon_W'] = float(stat)
            res['wilcoxon_p'] = float(p)
        except ValueError as e:
            res['wilcoxon_W'] = np.nan
            res['wilcoxon_p'] = np.nan
            res['note'] = str(e)

        # Smallest true effect a paired test on this many languages could detect at
        # alpha=0.05 two-sided with 80% power, given the observed spread. Anything smaller
        # than this is not something the current grid can distinguish from zero.
        if n >= 3 and res['sd_delta'] > 0:
            t_a = stats.t.ppf(0.975, n - 1)
            t_b = stats.t.ppf(0.80, n - 1)
            res['min_detectable_effect'] = float((t_a + t_b) * res['sd_delta'] / np.sqrt(n))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_file', type=str,
                   default=os.path.join('results_all', 'acc', 't1_sample_efficiency.csv'))
    p.add_argument('--output_file', type=str,
                   default=os.path.join('results_all', 'acc', 't2_region_match.csv'))
    p.add_argument('--stats_file', type=str,
                   default=os.path.join('results_all', 'acc', 't2_region_match_stats.csv'))
    p.add_argument('--metric', type=str, default='best_cer')
    p.add_argument('--accounting_file', type=str,
                   default=os.path.join('results_all', 'acc',
                                        't4_data_accounting_by_language.csv'),
                   help='Optional; adds the per-language data-accounting flag to the output.')
    args = p.parse_args()

    # `state == 'finished'` is not enough on its own. During a re-run window a cell holds two
    # runs in the same serial, and once the re-run finishes BOTH are finished -- so filtering on
    # state alone would resurrect the duplicate and hand the paired subtraction below two rows
    # per cell. t1 names the canonical run explicitly; use it.
    df = pd.read_csv(args.input_file)
    # t1 spans the grid and the control arms; a cross-language contrast is grid-only.
    if 'serial' in df.columns:
        df = df[df['serial'] == GRID_SERIAL]
    if 'is_canonical' in df.columns:
        df = df[df['is_canonical']]
    df = df[df['state'] == 'finished']
    df = df[df['model_short'].isin(CORE_VARIANTS)]
    assert_unique_keys(df, ['model_short', 'dataset'], label='t1 input to region match')

    # PRIMARY: failed runs removed. A language whose matched arm was the failed run cannot
    # contribute a matched-vs-mismatched delta and is dropped, with the reason on the row.
    primary = build_contrast(df, args.metric)

    # SENSITIVITY: nothing excluded, so the failed run is still counted against the
    # hypothesis. Reported alongside so the exclusion cannot quietly manufacture the result.
    df_all = df.copy()
    df_all['excluded_from_aggregate'] = False
    sensitivity = build_contrast(df_all, args.metric)

    primary['analysis'] = 'primary_excluding_failed_runs'
    sensitivity['analysis'] = 'sensitivity_including_all_runs'
    table = pd.concat([primary, sensitivity], ignore_index=True)

    # Carry the data-accounting verdict onto every language row, so a reader of t2 alone can
    # see which cells rest on a training stream that does not reconcile with its corpus.
    if os.path.exists(args.accounting_file):
        acc = pd.read_csv(args.accounting_file)[
            ['dataset', 'accounting_flag', 'ratio_implied_to_expected']]
        assert_unique_keys(acc, ['dataset'], label='t4 by-language join key')
        table = table.merge(acc, on='dataset', how='left')

    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    table.to_csv(args.output_file, index=False, float_format=FLOAT_FORMAT)
    print(f'Wrote {len(table)} rows to {args.output_file}')

    stats_rows = []
    for name, tab in (('primary_excluding_failed_runs', primary),
                      ('sensitivity_including_all_runs', sensitivity)):
        use = tab[tab['usable_primary']] if name.startswith('primary') else tab
        for col, label in (('delta_vs_mismatched', 'matched_vs_mismatched'),
                           ('delta_vs_global', 'matched_vs_global')):
            r = paired_test(use[col], label)
            r['analysis'] = name
            r['metric'] = args.metric
            stats_rows.append(r)

    sdf = pd.DataFrame(stats_rows)
    sdf.to_csv(args.stats_file, index=False, float_format=FLOAT_FORMAT)
    print(f'Wrote {len(sdf)} rows to {args.stats_file}\n')

    cols = ['analysis', 'contrast', 'n_languages', 'n_favouring_matched', 'mean_delta',
            'ci95_low', 'ci95_high', 'wilcoxon_p', 'min_detectable_effect']
    print(sdf[[c for c in cols if c in sdf.columns]].to_string(index=False))
    return 0


if __name__ == '__main__':
    main()
