"""Is the convergence point a property of the language or of the decoder?

Two questions the per-run t1 columns cannot answer on their own:

  cluster  do the four decoders on one language converge together?
  spread   is the between-language spread larger than the between-decoder spread?

Answered by a one-way variance decomposition of log10(convergence point) grouped by language.
ICC is the share of variance that is between-language: 1.0 means the language fixes the answer
and the decoder is irrelevant, 0.0 means language tells you nothing. The F test is the same
decomposition's significance, and the permutation test repeats it under shuffled language
labels so the ICC has a null to be read against.

log10 because the convergence point is a duration: a 40-step difference means something
different at 100 steps than at 800.

A spread only means something against the spread two runs of the SAME configuration show, so
the replicate serials supply a noise floor on the same statistic, via the same `curve_stats`.

Usage:
    python analyze_convergence_clustering.py \
        --input_file results_all/acc/t1_sample_efficiency.csv \
        --output_file results_all/acc/t10_convergence_clustering.csv
"""
import os
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from analyze_sample_efficiency import curve_stats
from utils import (LANGUAGE_DIC, RESOURCE_TIER, GRID_SERIAL, REPLICATE_SERIAL_PAIRS,
                   MODEL_SHORT, assert_unique_keys)

FLOAT_FORMAT = '%.6f'

# Evaluations are logged every 10 steps, so no convergence point is resolved finer than that.
EVAL_STEP_INTERVAL = 10

MEASURES = ('step_to_1.5x_best', 'step_to_best', 'audio_h_to_1.5x_best', 'epoch_at_1.5x_best')

# epoch is step * effective_batch / stream size to r = 0.998, and stream size is a property of
# the language. Its near-1.0 ICC is that denominator, not a fact about convergence.
TAUTOLOGICAL_MEASURES = ('epoch_at_1.5x_best',)

N_PERMUTATIONS = 10000
PERMUTATION_SEED = 0


def icc_oneway(y, groups):
    """Share of variance in `y` lying between `groups`, with the F test of the same split.

    ICC(1) on unbalanced groups. Returns nan for the ICC when a negative estimate arises
    (between-group variance below noise), which is a real answer, not a failure.
    """
    k, n = groups.nunique(), len(y)
    if k < 2 or n <= k:
        return np.nan, np.nan, np.nan
    means = y.groupby(groups).mean()
    sizes = y.groupby(groups).size()
    ss_between = float((sizes * (means - y.mean()) ** 2).sum())
    ss_within = float(((y - groups.map(means)) ** 2).sum())
    ms_between, ms_within = ss_between / (k - 1), ss_within / (n - k)
    if ms_within == 0:
        return np.nan, np.nan, np.nan
    n0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    icc = (ms_between - ms_within) / (ms_between + (n0 - 1) * ms_within)
    f = ms_between / ms_within
    return icc, f, float(stats.f.sf(f, k - 1, n - k))


def permuted_icc_p(y, groups, n_perm=N_PERMUTATIONS, seed=PERMUTATION_SEED):
    """How often a random regrouping clusters at least as tightly as the real languages do."""
    observed, _, _ = icc_oneway(y, groups)
    if not np.isfinite(observed):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    labels = groups.to_numpy()
    hits = 0
    for _ in range(n_perm):
        shuffled = pd.Series(rng.permutation(labels), index=y.index)
        null, _, _ = icc_oneway(y, shuffled)
        if np.isfinite(null) and null >= observed:
            hits += 1
    return observed, (hits + 1) / (n_perm + 1)


def replicate_spreads(history_files, measure):
    """max/min of `measure` over pairs of runs of the SAME configuration -- the noise floor."""
    frames = [pd.read_csv(f, low_memory=False) for f in history_files if os.path.exists(f)]
    if not frames:
        return pd.DataFrame(columns=['dataset', 'model_short', 'spread_ratio'])
    h = pd.concat(frames, ignore_index=True)

    recs = []
    for run_id, g in h.groupby('run_id'):
        ev = g[g['eval/cer'].notna()].sort_values('audio_hours')
        if ev.empty or g.iloc[0].get('state') != 'finished':
            continue
        first = g.iloc[0]
        recs.append({'serial': first.get('serial'), 'dataset': first.get('dataset'),
                     'model_short': MODEL_SHORT.get(first.get('model_id')),
                     **curve_stats(ev)})
    d = pd.DataFrame(recs)

    key = ['dataset', 'model_short']
    rows = []
    for canonical, superseded in REPLICATE_SERIAL_PAIRS:
        a = d[d['serial'] == canonical].drop_duplicates(key).set_index(key)
        b = d[d['serial'] == superseded].drop_duplicates(key).set_index(key)
        for cell in a.index.intersection(b.index):
            v = [a.loc[cell, measure], b.loc[cell, measure]]
            if not all(np.isfinite(v)) or min(v) == 0:
                continue
            rows.append({'dataset': cell[0], 'model_short': cell[1],
                         'serial_pair': f'{canonical}<->{superseded}',
                         'first': v[0], 'rerun': v[1],
                         'spread_ratio': max(v) / min(v)})
    return pd.DataFrame(rows)


def per_language(d, measure):
    """One row per language: where its decoders landed and how far apart."""
    rows = []
    for lang, g in d.groupby('dataset'):
        v = g[measure].dropna()
        if v.empty:
            continue
        # max/min is the readable form of "clustered": 1.0 is identical, 2.0 is twice as long.
        rows.append({
            'dataset': lang,
            'language_name': LANGUAGE_DIC.get(lang, lang),
            'resource_tier': RESOURCE_TIER.get(lang),
            'measure': measure,
            'n_models': int(len(v)),
            'min': v.min(),
            'median': v.median(),
            'max': v.max(),
            'spread_ratio': v.max() / v.min() if v.min() else np.nan,
            'cv_log10': float(np.std(np.log10(v), ddof=1)) if len(v) > 1 else np.nan,
            'fastest_model': g.loc[v.idxmin(), 'model_short'],
            'slowest_model': g.loc[v.idxmax(), 'model_short'],
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_file', type=str,
                   default=os.path.join('results_all', 'acc', 't1_sample_efficiency.csv'))
    p.add_argument('--output_file', type=str,
                   default=os.path.join('results_all', 'acc',
                                        't10_convergence_clustering.csv'))
    p.add_argument('--per_language_file', type=str,
                   default=os.path.join('results_all', 'acc',
                                        't10_convergence_by_language.csv'))
    p.add_argument('--history_files', type=str, nargs='+',
                   default=[os.path.join('data', 'raw_serials', f'history_serial_{s}.csv')
                            for s in (0, 1, 2, 3, 4)],
                   help='Serials holding replicate pairs, for the noise floor.')
    p.add_argument('--n_permutations', type=int, default=N_PERMUTATIONS)
    args = p.parse_args()

    d = pd.read_csv(args.input_file)
    # The grid only: control arms sit in serial 2 and would put 6 models on one language and 4
    # on the rest, which changes what the decomposition is comparing.
    d = d[(d['serial'] == GRID_SERIAL) & d['is_canonical'] & (d['state'] == 'finished')]
    assert_unique_keys(d, ['dataset', 'model_short'], label='t10 convergence grid')
    print(f'{len(d)} runs over {d["dataset"].nunique()} languages '
          f'x {d["model_short"].nunique()} decoders\n')

    stats_rows, lang_frames = [], []
    for measure in MEASURES:
        v = d[measure].dropna()
        sub = d[d[measure].notna()]
        y = np.log10(sub[measure].replace(0, np.nan)).dropna()
        groups = sub.loc[y.index, 'dataset']
        icc, f, f_p = icc_oneway(y, groups)
        _, perm_p = permuted_icc_p(y, groups, n_perm=args.n_permutations)

        # Decoder spread within a language, against the spread two runs of one configuration
        # show. If they match, the decoders are not ordered by convergence speed at all.
        lang = per_language(sub, measure)
        rep = replicate_spreads(args.history_files, measure)
        u_p = np.nan
        if len(rep) > 1 and len(lang) > 1:
            u_p = float(stats.mannwhitneyu(lang['spread_ratio'].dropna(),
                                           rep['spread_ratio'].dropna(),
                                           alternative='greater').pvalue)

        stats_rows.append({
            'measure': measure,
            'n_runs': int(len(y)),
            'n_languages': int(groups.nunique()),
            'overall_min': v.min(),
            'overall_median': v.median(),
            'overall_max': v.max(),
            'overall_spread_ratio': v.max() / v.min() if v.min() else np.nan,
            'icc_language': icc,
            'f_statistic': f,
            'f_p_value': f_p,
            'permutation_p': perm_p,
            'median_within_language_spread_ratio': lang['spread_ratio'].median(),
            'n_replicate_pairs': int(len(rep)),
            'median_replicate_spread_ratio': rep['spread_ratio'].median()
            if len(rep) else np.nan,
            'max_replicate_spread_ratio': rep['spread_ratio'].max() if len(rep) else np.nan,
            'decoder_spread_exceeds_noise_p': u_p,
            'icc_is_tautological': measure in TAUTOLOGICAL_MEASURES,
        })
        lang_frames.append(lang)

    out = pd.DataFrame(stats_rows)
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    out.to_csv(args.output_file, index=False, float_format=FLOAT_FORMAT)
    print(out.round(4).to_string(index=False))
    print(f'\nWrote {len(out)} rows to {args.output_file}')

    by_lang = pd.concat(lang_frames, ignore_index=True)
    by_lang.to_csv(args.per_language_file, index=False, float_format=FLOAT_FORMAT)
    print(f'Wrote {len(by_lang)} rows to {args.per_language_file}\n')

    head = MEASURES[0]
    show = by_lang[by_lang['measure'] == head].sort_values('median')
    print(f'{head}, per language:')
    print(show[['language_name', 'resource_tier', 'min', 'median', 'max',
                'spread_ratio', 'fastest_model']].round(2).to_string(index=False))

    row = out[out['measure'] == head].iloc[0]
    print(f'\nEvery run converges within {row["overall_min"]:.0f}-{row["overall_max"]:.0f} steps '
          f'({row["overall_spread_ratio"]:.1f}x end to end), and evaluations are only logged '
          f'every {EVAL_STEP_INTERVAL} steps.')
    print(f'Language explains {row["icc_language"]:.0%} of the variance in log10 '
          f'{head} (permutation p = {row["permutation_p"]:.4f}). The median language spans '
          f'{row["median_within_language_spread_ratio"]:.2f}x across its four decoders, against '
          f'{row["median_replicate_spread_ratio"]:.2f}x between two runs of ONE configuration '
          f'({int(row["n_replicate_pairs"])} pairs, one-sided '
          f'p = {row["decoder_spread_exceeds_noise_p"]:.3f}).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
