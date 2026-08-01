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
T5 = os.path.join(ACC, 't5_volume_interaction.csv')
T5S = os.path.join(ACC, 't5_volume_stats.csv')
T6A = os.path.join(ACC, 't6_loss_by_axis.csv')
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
    # The accounting numbers §4.1 actually prints. The per-language reconstruction table was
    # superseded by the volume table, so only the Tamil figures that carry the argument remain:
    # the reconstruction's accuracy (8,825 inferred vs 8,846 true) and the size of the loss.
    ('ta_in reconstructed stream',
     lambda: _acc('ta_in', 'implied_stream_samples'), 0),
    ('ta_in post-filter expected stream',
     lambda: _acc('ta_in', 'expected_stream_examples_post_filter'), 0),
    ('ta_in pre-filter expected stream',
     lambda: _acc('ta_in', 'expected_stream_examples'), 0),
    ('ta_in clips dropped by the cap',
     lambda: _acc('ta_in', 'n_dropped_by_cap'), 0),

    # The volume-interaction table and its statistics.
    *[(f'volume {lang} / {col}', (lambda lang=lang, col=col: _t5(lang, col)), nd)
      for lang in ('ta_in', 'ha_ng', 'mr_in', 'id_id', 'fr_fr', 'sw_ke', 'hi_in')
      for col, nd in (('stream_post_filter', 0), ('epochs_logged', 2),
                      ('delta_vs_mismatched', 2), ('baseline_cer', 2),
                      ('relative_delta_vs_mismatched_pct', 2))],
    *[(f'volume stats {sub} / {col}',
       (lambda sub=sub, col=col: _t5s(sub, 'log10_stream', 'delta_vs_mismatched', col)), nd)
      for sub in ('all_languages', 'excluding_ta_in')
      for col, nd in (('spearman_rho', 3), ('spearman_p', 4))],
    ('volume stats partial pearson_r',
     lambda: _t5s('all_languages', 'log10_stream|baseline_cer',
                  'delta_vs_mismatched', 'pearson_r'), 3),
    ('volume stats partial pearson_p',
     lambda: _t5s('all_languages', 'log10_stream|baseline_cer',
                  'delta_vs_mismatched', 'pearson_p'), 2),
    ('n_dropped_by_cap for ta_in', lambda: _t5('ta_in', 'n_dropped_by_cap'), 0),

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


def _axis(axis, level, col):
    d = load(T6A)
    sel = d[(d['axis'] == axis) & (d['level'] == level)]
    assert len(sel) == 1, f'{axis}/{level} matched {len(sel)} rows in {T6A}'
    return float(sel.iloc[0][col])


def _tier_monotone(col):
    """Tier order very_low -> low -> mid -> high must be non-increasing in `col`."""
    d = load(T6A)
    d = d[d['axis'] == 'resource_tier'].set_index('level')
    order = [t for t in ('very_low', 'low', 'mid', 'high') if t in d.index]
    vals = [float(d.loc[t, col]) for t in order]
    return all(a >= b - 1e-9 for a, b in zip(vals, vals[1:]))


def _t5(lang, col):
    d = load(T5)
    sel = d[d['dataset'] == lang]
    assert len(sel) == 1, f'{lang} matched {len(sel)} rows in {T5}'
    return sel.iloc[0][col]


def _t5s(subset, x, y, col):
    d = load(T5S)
    sel = d[(d['subset'] == subset) & (d['x'] == x) & (d['y'] == y)]
    assert len(sel) == 1, f'{subset}/{x}/{y} matched {len(sel)} rows in {T5S}'
    return float(sel.iloc[0][col])


def _monotone_except_one():
    """Delta should increase with stream size, with exactly one adjacent rank inversion."""
    d = load(T5).sort_values('stream_post_filter')
    vals = list(d['delta_vs_mismatched'])
    inversions = sum(1 for a, b in zip(vals, vals[1:]) if a > b)
    return inversions == 1


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
    # NOTE -- several claims here changed on 2026-08-01 when am_et and ur_pk were added, the
    # Spain-Spanish runs were replaced by es_mx, and the en_us/water re-run removed the last
    # exclusion. The old claims were not "fixed"; they were falsified by more data, and the
    # replacements below record what is true now. That is what this list is for.
    ('global is the slowest variant on sample efficiency',
     lambda: rank_means('audio_h_to_1.5x_best').idxmax() == 'global'),
    ('global is also the worst mean rank on best_cer',
     lambda: rank_means('best_cer').idxmax() == 'global'),
    ('the sample-efficiency and accuracy orderings still differ',
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
    ('median within-run noise exceeds the mean region effect',
     lambda: finished_core()['late_sd'].median()
     > abs(stat('sensitivity_including_all_runs', 'matched_vs_mismatched', 'mean_delta'))),
    # No runs are excluded any more: en_us/water replicated, so it is an effect, not a failure.
    ('no runs are excluded from aggregates',
     lambda: int(load(T1)['excluded_from_aggregate'].sum()) == 0),
    ('the primary and sensitivity analyses are now identical (nothing excluded)',
     lambda: stat('primary_excluding_failed_runs', 'matched_vs_mismatched', 'mean_delta')
     == stat('sensitivity_including_all_runs', 'matched_vs_mismatched', 'mean_delta')),
    ('best_cer <= final_cer for every run',
     lambda: bool((load(T1)['best_cer'] <= load(T1)['final_cer'] + 1e-9).all())),
    ('all four regional/global variants share one parameter count',
     lambda: _regional_params() > 0),

    # --- data accounting -------------------------------------------------------------
    ('no language fails to reconcile post-filter',
     lambda: load(T4L).query(
         "accounting_flag == 'does_not_reconcile_see_docstring'").empty),
    ('ta_in reconciles once the cap loss is accounted for',
     lambda: abs(float(_acc('ta_in', 'ratio_implied_to_expected_post_filter')) - 1.0) < 0.05),
    ('ta_in did NOT reconcile against the pre-filter count (the loss is real)',
     lambda: float(_acc('ta_in', 'ratio_implied_to_expected')) < 0.5),
    ('only ta_in and fr_fr lose clips to the cap',
     lambda: set(load(T4L).query("n_dropped_by_cap > 0")['dataset']) == {'ta_in', 'fr_fr'}),
    ('en_us and hi_in never exhausted their streams, so are lower bounds only',
     lambda: set(load(T4L).query(
         "accounting_flag == 'stream_never_exhausted_lower_bound_only'")['dataset'])
     == {'en_us', 'hi_in'}),

    # --- the volume interaction, as it now stands ------------------------------------
    # These are the claims that changed most, and the honest versions are weaker.
    ('the volume correlation is still significant over all languages',
     lambda: (_t5s('all_languages', 'log10_stream', 'delta_vs_mismatched',
                   'spearman_rho') > 0.5
              and _t5s('all_languages', 'log10_stream', 'delta_vs_mismatched',
                       'spearman_p') < 0.05)),
    ('but it NO LONGER survives dropping the extreme point -- it did with 7 languages',
     lambda: _t5s('excluding_ta_in', 'log10_stream', 'delta_vs_mismatched',
                  'spearman_p') > 0.05),
    ('am_et and ta_in sit at the same volume with OPPOSITE signs -- the key new evidence',
     lambda: (lambda d: (abs(d.loc['am_et', 'stream_post_filter']
                             - d.loc['ta_in', 'stream_post_filter']) < 500
                         and d.loc['am_et', 'delta_vs_mismatched'] > 0
                         > d.loc['ta_in', 'delta_vs_mismatched']))(
         load(T5).set_index('dataset'))),
    ('volume and baseline CER are collinear, so the explanations stay entangled',
     lambda: _t5s('all_languages', 'log10_stream', 'baseline_cer', 'spearman_rho') < -0.3),

    # --- loss diagnostics ------------------------------------------------------------
    ('overfitting (eval-loss rise) is monotone in resource tier',
     lambda: _tier_monotone('median_eval_loss_rise')),
    ('low-resource cells reach their best eval loss far earlier in the run',
     lambda: _axis('resource_tier', 'very_low', 'mean_frac_to_best')
     < _axis('resource_tier', 'high', 'mean_frac_to_best')),
    ('cross-domain evals have a larger generalisation gap than in-domain',
     lambda: _axis('eval_domain', 'cross_domain', 'median_generalisation_gap')
     > _axis('eval_domain', 'in_domain', 'median_generalisation_gap')),
    ('but domain shift does NOT show up as overfitting -- the two are separable',
     lambda: _axis('eval_domain', 'cross_domain', 'median_eval_loss_rise') < 0.05),
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
