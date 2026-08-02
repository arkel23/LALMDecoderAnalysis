"""
How large is "regional specialisation", actually? And does the size of it predict anything?

The study treats specialisation as a categorical label -- earth / fire / water -- and asks
whether the matched label beats the mismatched ones. That framing cannot distinguish "regional
specialisation does not help" from "the specialisation was too small to measure". The Tiny Aya
report's Appendix A tables settle that, because they give the actual share of each variant's
post-training mix that was in each language (fetch_tinyaya_composition.py).

Two quantities per language:

  excess_pp   matched-variant exposure minus the mean exposure of the two mismatched variants,
              in percentage points. This is the SIZE OF THE TREATMENT -- how much more of the
              target language the "matched" decoder actually saw.
  ratio_m_over_g   matched exposure divided by global exposure, i.e. how much specialisation
              buys over the non-regional model.

The headline is the treatment size, not the correlation: if the median excess is ~1 percentage
point of post-training data, then a null region-match result is not surprising and should not
be reported as though a strong manipulation failed.

One language inverts the design and is worth reading carefully rather than dropping. English's
matched variant is `water` at 17.0 % English, but the `fire` mix is 46.2 % English -- so for
English the "matched" decoder is actually the LESS exposed one, `excess_pp` is negative, and
the observed delta is positive (matched worse). That is the exposure account making a correct
prediction on the one cell where exposure and label disagree.

Usage:
    python analyze_exposure.py --output_file results_all/acc/t8_exposure.csv
"""
import os
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from utils import load_tinyaya_composition, LANGUAGE_REGION, LANGUAGE_DIC, RESOURCE_TIER

REGIONAL = ('earth', 'fire', 'water')
FLOAT_FORMAT = '%.6f'


def build_table(volume_file):
    comp = load_tinyaya_composition()
    if comp is None:
        raise SystemExit('Run fetch_tinyaya_composition.py first.')
    comp = comp.set_index('dataset')
    vol = pd.read_csv(volume_file)

    rows = []
    for _, r in vol.iterrows():
        lang = r['dataset']
        region = LANGUAGE_REGION.get(lang)
        if lang not in comp.index or region is None:
            continue
        matched = float(comp.loc[lang, region])
        glob = float(comp.loc[lang, 'global'])
        mismatched = float(np.mean([comp.loc[lang, v] for v in REGIONAL if v != region]))
        rows.append({
            'dataset': lang,
            'language_name': LANGUAGE_DIC.get(lang),
            'region': region,
            'resource_tier': RESOURCE_TIER.get(lang),
            'exposure_matched_pct': matched,
            'exposure_global_pct': glob,
            'exposure_mismatched_mean_pct': mismatched,
            'excess_pp': matched - mismatched,
            'ratio_matched_over_global': matched / glob if glob else np.nan,
            'delta_vs_mismatched': r['delta_vs_mismatched'],
            'delta_vs_global': r['delta_vs_global'],
        })
    return pd.DataFrame(rows).sort_values('excess_pp')


def stats_rows(df):
    out = []
    for xname in ('excess_pp', 'ratio_matched_over_global'):
        for yname in ('delta_vs_mismatched', 'delta_vs_global'):
            x, y = df[xname], df[yname]
            rho, p_rho = stats.spearmanr(x, y)
            r, p_r = stats.pearsonr(x, y)
            out.append({'x': xname, 'y': yname, 'n': len(df),
                        'spearman_rho': rho, 'spearman_p': p_rho,
                        'pearson_r': r, 'pearson_p': p_r})
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--volume_file', type=str,
                   default=os.path.join('results_all', 'acc', 't5_volume_interaction.csv'))
    p.add_argument('--output_file', type=str,
                   default=os.path.join('results_all', 'acc', 't8_exposure.csv'))
    p.add_argument('--stats_file', type=str,
                   default=os.path.join('results_all', 'acc', 't8_exposure_stats.csv'))
    args = p.parse_args()

    df = build_table(args.volume_file)
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    df.to_csv(args.output_file, index=False, float_format=FLOAT_FORMAT)
    print(f'Wrote {len(df)} rows to {args.output_file}\n')
    print(df[['dataset', 'region', 'resource_tier', 'exposure_matched_pct',
              'exposure_global_pct', 'exposure_mismatched_mean_pct', 'excess_pp',
              'delta_vs_mismatched']].round(3).to_string(index=False))

    sdf = stats_rows(df)
    sdf.to_csv(args.stats_file, index=False, float_format=FLOAT_FORMAT)
    print(f'\nWrote {len(sdf)} rows to {args.stats_file}')
    print(sdf.round(4).to_string(index=False))

    median_excess = df['excess_pp'].median()
    n_small = int((df['excess_pp'] < 1.0).sum())
    print(f'\nTREATMENT SIZE: the matched variant saw a median of {median_excess:.2f} '
          f'percentage points more of the target language than the mismatched variants did.')
    print(f'{n_small}/{len(df)} languages had less than 1 pp of separation.')
    print('A null region-match result against a ~1 pp manipulation is a weak-treatment null, '
          'not a demonstration that specialisation cannot matter.')
    return 0


if __name__ == '__main__':
    main()
