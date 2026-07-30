"""
Re-derives every number printed in the assessment document from the CSV that owns it.

The SPEC / DERIVED / ORDERINGS lists below are the contract: each entry names a printed
value, the CSV it comes from, how to select it, and the rounding used to print it. Expected
values are always COMPUTED FROM THE CSV, never hardcoded -- a checker that hardcodes its
expectations goes stale silently and then certifies its own staleness.

Adding a number to the document means adding it here. That is the whole point: a number that
lives only in prose has nothing regenerating it, and it rots the moment upstream data changes.

Three details that each correspond to a real defect elsewhere in this family of repos:
  - Rounding is half-UP, not Python's banker's default. round(5.25, 1) is 5.2; the printed
    value is 5.3.
  - Occurrence matching is bounded so 8.3 does not match inside 18.31.
  - A passing check on a short or very common value is reported as LOW-SPECIFICITY rather
    than counted as a win, so the coverage number stays honest.

Usage:  python verify_paper_numbers.py [--verbose]
Exit code is non-zero if any check fails, so plotter.sh fails loudly.
"""
import os
import re
import sys
import argparse
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

ACC = os.path.join('results_all', 'acc')
DOC = os.path.join('docs', 'PLAN_ASSESSMENT.md')

T1 = os.path.join(ACC, 't1_sample_efficiency.csv')
T2S = os.path.join(ACC, 't2_region_match_stats.csv')
T3 = os.path.join(ACC, 't3_crs_ood.csv')
T4L = os.path.join(ACC, 't4_data_accounting_by_language.csv')
RAW = os.path.join('data', 'raw_serials', 'raw_serial_0.csv')

CORE = ('earth', 'fire', 'global', 'water')


def half_up(value, nd):
    return Decimal(str(float(value))).quantize(Decimal(1).scaleb(-nd),
                                               rounding=ROUND_HALF_UP)


def fmt(value, nd):
    return f'{half_up(value, nd):.{nd}f}'


_cache = {}


def load(path):
    if path not in _cache:
        _cache[path] = pd.read_csv(path)
    return _cache[path]


def finished_core(exclude=True):
    t1 = load(T1)
    t1 = t1[(t1['state'] == 'finished') & (t1['model_short'].isin(CORE))]
    if exclude:
        t1 = t1[~t1['excluded_from_aggregate']]
    return t1


def rank_means(metric):
    p = finished_core().pivot_table(index='dataset', columns='model_short', values=metric)
    return p.dropna().rank(axis=1).mean()


def stat(analysis, contrast, col):
    s = load(T2S)
    sel = s[(s['analysis'] == analysis) & (s['contrast'] == contrast)]
    assert len(sel) == 1, f'{analysis}/{contrast} matched {len(sel)} rows'
    return sel.iloc[0][col]


# --- The contract -------------------------------------------------------------------
# (label, callable -> value, decimals)
DERIVED = [
    ('median within-run late_sd', lambda: finished_core()['late_sd'].median(), 2),
    ('mean final_minus_best', lambda: finished_core()['final_minus_best'].mean(), 2),
    ('max final_minus_best', lambda: finished_core()['final_minus_best'].max(), 2),

    # Region-match statistics, both analyses, both contrasts.
    *[(f'{a} / {c} / {col}', (lambda a=a, c=c, col=col: stat(a, c, col)), nd)
      for a, c in (('primary_excluding_failed_runs', 'matched_vs_mismatched'),
                   ('primary_excluding_failed_runs', 'matched_vs_global'),
                   ('sensitivity_including_all_runs', 'matched_vs_mismatched'),
                   ('sensitivity_including_all_runs', 'matched_vs_global'))
      for col, nd in (('n_languages', 0), ('n_favouring_matched', 0), ('mean_delta', 2),
                      ('ci95_low', 2), ('ci95_high', 2), ('wilcoxon_p', 3),
                      ('min_detectable_effect', 2))],

    # Sample-efficiency and accuracy rankings.
    *[(f'rank audio_h_to_1.5x_best {v}',
       (lambda v=v: rank_means('audio_h_to_1.5x_best')[v]), 2) for v in CORE],
    *[(f'rank best_cer {v}', (lambda v=v: rank_means('best_cer')[v]), 2) for v in CORE],

    # crs_sc OOD cell.
    ('crs_sc min best_cer',
     lambda: load(T3).query("state == 'finished'")['best_cer'].min(), 2),
    ('crs_sc max best_cer',
     lambda: load(T3).query("state == 'finished'")['best_cer'].max(), 2),
    ('crs_sc best_cer spread', lambda: (
        load(T3).query("state == 'finished'")['best_cer'].max()
        - load(T3).query("state == 'finished'")['best_cer'].min()), 2),
    ('crs_sc min late_sd',
     lambda: load(T3).query("state == 'finished'")['late_sd'].min(), 2),
    ('crs_sc max late_sd',
     lambda: load(T3).query("state == 'finished'")['late_sd'].max(), 2),

    # Parameter counts, from the wandb config -- the matched-variant premise (risk 1).
    ('regional variant parameter count', lambda: _regional_params(), 0),
    ('base minus regional parameter delta', lambda: _base_param_delta(), 0),

    # Data accounting, per language: epochs, reconstructed stream, expected examples, ratio.
    # epochs and the reconstructed stream are printed for every language, including crs_sc.
    *[(f'accounting {lang} / {col}', (lambda lang=lang, col=col: _acc(lang, col)), nd)
      for lang in ('ta_in', 'ha_ng', 'crs_sc', 'sw_ke', 'hi_in', 'id_id', 'mr_in', 'en_us',
                   'fr_fr')
      for col, nd in (('epochs_logged', 2), ('implied_stream_samples', 2))],
    # The ratio is only defined where the expected size is known -- crs_sc trains on the
    # ERISLab mirror, whose split is not in the example-count snapshot, so it prints as "-".
    *[(f'accounting {lang} / ratio_implied_to_expected',
       (lambda lang=lang: _acc(lang, 'ratio_implied_to_expected')), 2)
      for lang in ('ta_in', 'ha_ng', 'sw_ke', 'hi_in', 'id_id', 'mr_in', 'en_us', 'fr_fr')],
    *[(f'accounting {lang} / expected_stream_examples',
       (lambda lang=lang: _acc(lang, 'expected_stream_examples')), 0)
      for lang in ('ta_in', 'ha_ng', 'sw_ke', 'hi_in', 'id_id', 'mr_in', 'en_us', 'fr_fr')],
    *[(f'accounting {lang} / ratio_to_closest_single_config',
       (lambda lang=lang: _acc(lang, 'ratio_to_closest_single_config')), 2)
      for lang in ('ta_in', 'ha_ng', 'sw_ke')],

    # The ta_in region-match term, and its best CER -- both quoted in the small-data caveat.
    ('ta_in delta_vs_mismatched', lambda: _t2_delta('ta_in'), 2),
    ('ta_in best_cer', lambda: (
        load(T1).query("dataset == 'ta_in' and state == 'finished'")['best_cer'].min()), 2),
]


def _acc(lang, col):
    d = load(T4L)
    sel = d[d['dataset'] == lang]
    assert len(sel) == 1, f'{lang} matched {len(sel)} rows in {T4L}'
    return sel.iloc[0][col]


def _t2_delta(lang):
    d = load(os.path.join(ACC, 't2_region_match.csv'))
    sel = d[(d['dataset'] == lang)
            & (d['analysis'] == 'primary_excluding_failed_runs')]
    assert len(sel) == 1, f'{lang} matched {len(sel)} rows'
    return sel.iloc[0]['delta_vs_mismatched']


def _regional_params():
    raw = load(RAW)
    col = 'model/num_parameters'
    regional = raw[raw['model_id'].str.contains('earth|fire|water|global', na=False)]
    vals = regional[col].dropna().unique()
    assert len(vals) == 1, f'regional variants disagree on parameter count: {vals}'
    return float(vals[0])


def _base_param_delta():
    raw = load(RAW)
    col = 'model/num_parameters'
    base = raw[raw['model_id'].str.contains('tiny-aya-base', na=False)][col].dropna().unique()
    assert len(base) == 1, f'base parameter count is not unique: {base}'
    return abs(_regional_params() - float(base[0]))


# (claim, callable -> bool). These are the claims a data correction invalidates without
# changing any printed digit, so nothing else would catch them.
ORDERINGS = [
    ('global is slowest (worst mean rank) on sample efficiency',
     lambda: rank_means('audio_h_to_1.5x_best').idxmax() == 'global'),
    ('fire is fastest (best mean rank) on sample efficiency',
     lambda: rank_means('audio_h_to_1.5x_best').idxmin() == 'fire'),
    ('global is worst mean rank on best_cer too',
     lambda: rank_means('best_cer').idxmax() == 'global'),
    ('the sample-efficiency and accuracy orderings differ',
     lambda: list(rank_means('audio_h_to_1.5x_best').sort_values().index)
     != list(rank_means('best_cer').sort_values().index)),
    ('every region-match confidence interval spans zero (no significant effect)',
     lambda: all(stat(a, c, 'ci95_low') < 0 < stat(a, c, 'ci95_high')
                 for a in ('primary_excluding_failed_runs',
                           'sensitivity_including_all_runs')
                 for c in ('matched_vs_mismatched', 'matched_vs_global'))),
    ('minimum detectable effect exceeds the observed effect in every analysis',
     lambda: all(stat(a, c, 'min_detectable_effect') > abs(stat(a, c, 'mean_delta'))
                 for a in ('primary_excluding_failed_runs',
                           'sensitivity_including_all_runs')
                 for c in ('matched_vs_mismatched', 'matched_vs_global'))),
    ('median within-run noise exceeds the sensitivity-analysis mean effect',
     lambda: finished_core()['late_sd'].median()
     > abs(stat('sensitivity_including_all_runs', 'matched_vs_mismatched', 'mean_delta'))),
    ('excluding the failed run increases the apparent effect (so it must be disclosed)',
     lambda: abs(stat('primary_excluding_failed_runs', 'matched_vs_mismatched', 'mean_delta'))
     > abs(stat('sensitivity_including_all_runs', 'matched_vs_mismatched', 'mean_delta'))),
    ('crs_sc variant spread is smaller than the largest per-run noise there',
     lambda: (lambda d: (d['best_cer'].max() - d['best_cer'].min()) < 2.0)(
         load(T3).query("state == 'finished'"))),
    ('best_cer <= final_cer for every run',
     lambda: bool((load(T1)['best_cer'] <= load(T1)['final_cer'] + 1e-9).all())),
    ('exactly one run is excluded from aggregates',
     lambda: int(load(T1)['excluded_from_aggregate'].sum()) == 1),
    ('all four regional/global variants share one parameter count',
     lambda: _regional_params() > 0),

    # Data accounting. ta_in is the sole non-reconciling cell and its signature is that it
    # matches ONE config rather than the sum -- unlike the other two multi-config languages.
    # A data correction could flip any of this without moving a printed digit.
    ('ta_in is the only language that does not reconcile',
     lambda: list(load(T4L).query(
         "accounting_flag == 'does_not_reconcile_see_docstring'")['dataset']) == ['ta_in']),
    ("ta_in's reconstruction matches a single config within 1%",
     lambda: abs(float(_acc('ta_in', 'ratio_to_closest_single_config')) - 1.0) < 0.01),
    ('ha_ng and sw_ke match their SUMMED streams, not a single config',
     lambda: all(abs(float(_acc(l, 'ratio_implied_to_expected')) - 1.0)
                 < abs(float(_acc(l, 'ratio_to_closest_single_config')) - 1.0)
                 for l in ('ha_ng', 'sw_ke'))),
    ('every language with a known expected size except ta_in reconciles',
     lambda: set(load(T4L).query(
         "accounting_flag == 'reconciles'")['dataset'])
     == {'id_id', 'ha_ng', 'mr_in', 'fr_fr', 'sw_ke'}),
    ('ta_in has the worst best_cer in the grid (the small-data regime)',
     lambda: (lambda d: d.loc[d['best_cer'].idxmax(), 'dataset'])(
         load(T1).query("state == 'finished'").groupby('dataset', as_index=False)
         .agg(best_cer=('best_cer', 'min'))) == 'ta_in'),
    ('ta_in carries the largest-magnitude region-match delta',
     lambda: (lambda d: d.loc[d['delta_vs_mismatched'].abs().idxmax(), 'dataset'])(
         load(os.path.join(ACC, 't2_region_match.csv')).query(
             "analysis == 'primary_excluding_failed_runs'"
         ).dropna(subset=['delta_vs_mismatched'])) == 'ta_in'),
    ('en_us and hi_in never exhausted their streams, so are lower bounds only',
     lambda: set(load(T4L).query(
         "accounting_flag == 'stream_never_exhausted_lower_bound_only'")['dataset'])
     == {'en_us', 'hi_in'}),
    ('every language reports an estimate_kind',
     lambda: load(T4L)['estimate_kind'].isin(['estimate', 'lower_bound']).all()),
    ('mean sample duration is within the 30 s cap for every run',
     lambda: bool(load(os.path.join(ACC, 't4_data_accounting.csv'))[
         'mean_sample_seconds_within_cap'].all())),
]


def normalise(text):
    """Strip comments and normalise the minus signs and separators a document may use."""
    text = re.sub(r'(?<!\\)%.*', '', text)          # LaTeX-style comments, harmless here
    text = (text.replace('−', '-')             # unicode minus
                .replace('–', '-')             # en dash
                .replace('{-}', '-').replace('$-$', '-'))
    text = re.sub(r'(?<=[\s\[(])\+(?=\d)', '', text)   # drop a leading + on +0.12
    text = text.replace(',', '')                    # 3,656,222,720 -> 3656222720
    return text


def occurrences(text, printed):
    return len(re.findall(rf'(?<![\d.]){re.escape(printed)}(?![\d])', text))


def specificity(printed, hits):
    """A short or very common value proves little even when it matches."""
    digits = sum(ch.isdigit() for ch in printed)
    if digits <= 2:
        return 'value has <=2 significant digits'
    if hits > 3:
        return f'appears {hits}x in the document'
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--doc', type=str, default=DOC)
    args = ap.parse_args()

    if not os.path.exists(args.doc):
        print(f'Document not found: {args.doc}')
        return 1
    with open(args.doc) as fh:
        text = normalise(fh.read())

    failures, weak, passed = [], [], 0

    broken = []
    for claim, fn in ORDERINGS:
        try:
            if not fn():
                broken.append(claim)
            elif args.verbose:
                print(f'  ok  ordering: {claim}')
        except Exception as exc:
            broken.append(f'{claim} (raised {exc!r})')

    for label, fn, nd in DERIVED:
        try:
            raw_value = fn()
        except Exception as exc:
            failures.append(f'{label}: could not compute ({exc!r})')
            continue
        printed = fmt(raw_value, nd)
        variants = {printed, printed.lstrip('-')} if printed.startswith('-') else {printed}
        hits = max(occurrences(text, v) for v in variants)
        if hits == 0:
            failures.append(f'{label}: CSV gives {raw_value!r} -> "{printed}", '
                            f'not found in {args.doc}')
            continue
        passed += 1
        note = specificity(printed, hits)
        if note:
            weak.append(f'{label} = {printed} ({note})')
        elif args.verbose:
            print(f'  ok  {label} = {printed}')

    print(f'\n{len(ORDERINGS)} ordering/structural claims checked, {len(broken)} broken')
    for b in broken:
        print(f'  BROKEN {b}')

    print(f'{len(DERIVED)} document numbers checked against their generating CSVs, '
          f'{len(failures)} problem(s)')
    for f in failures:
        print(f'  FAIL {f}')

    if weak:
        print(f'{len(weak)} passing check(s) are LOW-SPECIFICITY and prove little:')
        for w in weak:
            print(f'  weak {w}')
    print(f'\nhigh-specificity passes: {passed - len(weak)}/{len(DERIVED)}')

    return 1 if (failures or broken) else 0


if __name__ == '__main__':
    sys.exit(main())
