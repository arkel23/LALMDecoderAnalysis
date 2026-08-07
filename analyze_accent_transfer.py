"""Does a model's in-domain or FLEURS score predict how it does on held-out varieties?

One row per (model, variety), with that SAME model's two anchors for the parent language:
`primary_wer` (its in-domain point) and `fleurs_wer` (its cross-domain point). Three outcomes --
the variety's own WER, the absolute penalty against the anchor, and the relative penalty.

Two things this script exists to keep straight, both of which change the answer:

  seen vs held-out  `in_domain_role == 'accent_transfer'` means "not the cell's primary point",
                    which includes sw_tz, ur_in and ha_td -- training data. `held_out_only` is
                    the headline population; `all_varieties` is reported beside it.
  selection source  `fleurs_wer` is downstream of selection for the ten cells chosen on FLEURS
                    validation, and zero-shot for ha_ng and crs_sc, which were chosen in-domain.
                    Carried as `parent_selected_on_fleurs`. Neither cell contributes a held-out
                    variety today, so it does not move the numbers -- it guards the moment one
                    does. Hausa's anchors are WorldSpeech ha_td in-domain, FLEURS ha_ng out.
  the language      Pooled, in-domain WER predicts variety WER at rho ~ .56. Centre both axes
                    within language and it goes to ~ .12. Most of the pooled figure is "hard
                    languages are hard for everyone", so every correlation is emitted at all
                    three aggregations and the gap is the finding.

Spearman throughout: one degenerate cell (en_pk) carries a ~285 WER residual that would drive
any Pearson estimate on its own. It is flagged, not dropped, and the headline is reported with
and without it.

Usage:
    python analyze_accent_transfer.py \
        --input_file results_all/acc/t7_baselines.csv \
        --output_file results_all/acc/t11_accent_transfer.csv
"""
import os
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from utils import (LANGUAGE_DIC, MODEL_SHORT, get_accent_match, in_domain_role,
                   is_trained_variety, selected_on_fleurs, to_study_cell, parent_model_id,
                   assert_unique_keys)

FLOAT_FORMAT = '%.6f'

# Above this the model has stopped transcribing and started generating; the row is kept and
# flagged so the headline can be reported with and without it.
DEGENERATE_WER = 150.0

# Below this a within-language correlation is not worth reporting as a number.
MIN_VARIETIES_FOR_WITHIN_LANGUAGE = 3

PREDICTORS = ('primary_wer', 'fleurs_wer')
OUTCOMES = ('variety_wer', 'penalty_rel')


def load_trained(path):
    """Serial 11, reduced to parent model ids so it shares an axis with the baselines."""
    if not os.path.exists(path):
        print(f'[SKIP] serial 11: {path} not present')
        return None
    df = pd.read_csv(path)
    df = df[(df['state'] == 'finished') & df['wer'].notna()].copy()
    df['model_short'] = df['model_id'].map(parent_model_id).map(MODEL_SHORT)
    return df


def annotate(df):
    """Add the study cell, eval domain and role columns the baselines table already carries."""
    out = df.copy()
    out['study_cell'] = out['dataset'].map(to_study_cell)
    out['eval_domain'] = np.where(
        out['dataset_path'].astype(str).str.contains('WorldSpeech'), 'in_domain', 'cross_domain')
    out['in_domain_role'] = [in_domain_role(c, d, e) for c, d, e
                             in zip(out['study_cell'], out['dataset'], out['eval_domain'])]
    return out


def build_rows(df, population):
    """One row per (model, variety), joined to the same model's two anchors."""
    key = ['model_short', 'study_cell']
    anchors = {}
    for name, mask in (('primary_wer', (df['in_domain_role'] == 'primary')),
                       ('fleurs_wer', (df['eval_domain'] == 'cross_domain'))):
        a = df[mask][key + ['wer']].rename(columns={'wer': name})
        assert_unique_keys(a, key, label=f'{population} anchor {name}')
        anchors[name] = a

    acc = df[df['in_domain_role'] == 'accent_transfer'][
        key + ['dataset', 'wer', 'num_samples']].rename(columns={'wer': 'variety_wer'})
    assert_unique_keys(acc, key + ['dataset'], label=f'{population} accented rows')

    for name, a in anchors.items():
        acc = acc.merge(a, on=key, how='left', validate='many_to_one')

    acc['population'] = population
    acc['language_name'] = acc['study_cell'].map(LANGUAGE_DIC)
    acc['accent_match'] = acc['study_cell'].map(get_accent_match)
    # The whole point of the split: sw_tz and ur_in are training data.
    acc['variety_seen_in_training'] = acc['dataset'].map(is_trained_variety)
    acc['parent_selected_on_fleurs'] = acc['study_cell'].map(selected_on_fleurs)
    acc['is_degenerate'] = acc['variety_wer'] > DEGENERATE_WER
    acc['penalty_abs'] = acc['variety_wer'] - acc['primary_wer']
    acc['penalty_rel'] = acc['variety_wer'] / acc['primary_wer']
    return acc.sort_values(['population', 'study_cell', 'dataset', 'model_short'])


def correlate(d, predictor, outcome, aggregation):
    """Spearman over the requested aggregation. Returns (rho, p, n)."""
    s = d[[predictor, outcome, 'study_cell']].replace([np.inf, -np.inf], np.nan).dropna()
    if aggregation == 'language_centred':
        for col in (predictor, outcome):
            s[col] = s[col] - s.groupby('study_cell')[col].transform('mean')
    if len(s) < 3 or s[predictor].nunique() < 2 or s[outcome].nunique() < 2:
        return np.nan, np.nan, len(s)
    r = stats.spearmanr(s[predictor], s[outcome])
    return float(r.statistic), float(r.pvalue), len(s)


def correlation_table(rows):
    """Every (predictor, outcome, population, aggregation) cell, flattering or not."""
    out = []
    for pop_label, pop in (('held_out_only', rows[~rows['variety_seen_in_training']]),
                           ('all_varieties', rows)):
        for drop_deg in (False, True):
            sub = pop[~pop['is_degenerate']] if drop_deg else pop
            for predictor in PREDICTORS:
                for outcome in OUTCOMES:
                    for agg in ('pooled', 'language_centred'):
                        rho, p, n = correlate(sub, predictor, outcome, agg)
                        out.append({'population': pop_label, 'excludes_degenerate': drop_deg,
                                    'predictor': predictor, 'outcome': outcome,
                                    'aggregation': agg, 'study_cell': 'ALL',
                                    'spearman_rho': rho, 'spearman_p': p, 'n': n})
                    # Within-language, where a language has enough varieties to support one.
                    for cell, g in sub.groupby('study_cell'):
                        if g['dataset'].nunique() < MIN_VARIETIES_FOR_WITHIN_LANGUAGE:
                            continue
                        rho, p, n = correlate(g, predictor, outcome, 'pooled')
                        out.append({'population': pop_label, 'excludes_degenerate': drop_deg,
                                    'predictor': predictor, 'outcome': outcome,
                                    'aggregation': 'within_language', 'study_cell': cell,
                                    'spearman_rho': rho, 'spearman_p': p, 'n': n})
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_file', type=str,
                   default=os.path.join('results_all', 'acc', 't7_baselines.csv'))
    p.add_argument('--trained_file', type=str,
                   default=os.path.join('data', 'raw_serials', 'raw_serial_11.csv'))
    p.add_argument('--output_file', type=str,
                   default=os.path.join('results_all', 'acc', 't11_accent_transfer.csv'))
    p.add_argument('--correlation_file', type=str,
                   default=os.path.join('results_all', 'acc', 't11_accent_correlations.csv'))
    p.add_argument('--per_language_file', type=str,
                   default=os.path.join('results_all', 'acc', 't11_accent_by_language.csv'))
    args = p.parse_args()

    base = pd.read_csv(args.input_file)
    base = base[(base['state'] == 'finished') & base['wer'].notna()]
    frames = [build_rows(base, 'baseline')]

    trained = load_trained(args.trained_file)
    if trained is not None:
        frames.append(build_rows(annotate(trained), 'trained'))

    rows = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    rows.to_csv(args.output_file, index=False, float_format=FLOAT_FORMAT)
    print(f'Wrote {len(rows)} rows to {args.output_file}\n')

    for pop, g in rows.groupby('population'):
        miss = int(g['primary_wer'].isna().sum() + g['fleurs_wer'].isna().sum())
        print(f'{pop}: {len(g)} (model, variety) rows over {g["dataset"].nunique()} varieties '
              f'x {g["model_short"].nunique()} models, {miss} missing anchor value(s)')
        print(f'  models: {", ".join(sorted(g["model_short"].dropna().unique()))}')
    seen = sorted(rows[rows['variety_seen_in_training']]['dataset'].unique())
    print(f'\nvarieties that were TRAINING data, not held out: {", ".join(seen)}')

    corr = pd.concat([correlation_table(g).assign(model_population=pop)
                      for pop, g in rows.groupby('population')], ignore_index=True)
    corr.to_csv(args.correlation_file, index=False, float_format=FLOAT_FORMAT)
    print(f'\nWrote {len(corr)} rows to {args.correlation_file}')
    head = corr[(corr['population'] == 'held_out_only') & (~corr['excludes_degenerate'])
                & (corr['outcome'] == 'variety_wer') & (corr['aggregation'] != 'within_language')]
    print(head[['model_population', 'predictor', 'aggregation', 'spearman_rho', 'spearman_p',
                'n']].round(4).to_string(index=False))
    off = sorted(rows.loc[~rows['parent_selected_on_fleurs'], 'study_cell'].unique())
    print(f'\ncells NOT selected on FLEURS, so whose FLEURS score is zero-shot w.r.t. '
          f'selection: {", ".join(off) if off else "none in this population"}')

    by_lang = (rows.groupby(['population', 'study_cell', 'dataset'], as_index=False)
               .agg(language_name=('language_name', 'first'),
                    seen_in_training=('variety_seen_in_training', 'first'),
                    n_models=('model_short', 'nunique'),
                    num_samples=('num_samples', 'first'),
                    median_variety_wer=('variety_wer', 'median'),
                    median_penalty_abs=('penalty_abs', 'median'),
                    median_penalty_rel=('penalty_rel', 'median'))
               .sort_values(['population', 'study_cell', 'dataset']))
    by_lang.to_csv(args.per_language_file, index=False, float_format=FLOAT_FORMAT)
    print(f'\nWrote {len(by_lang)} rows to {args.per_language_file}')
    print(by_lang[by_lang['population'] == 'baseline'].round(2).to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
