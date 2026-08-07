"""Logic tests for utils.py and the analysis scripts.

Not pytest: it is not in this repo's conda env, and adding a dependency would break the
bare-checkout bar.

Division of labour with verify_paper_numbers.py: that asks "does the document match the CSVs?",
this asks "is the logic right?". A number-checker cannot catch a CSV that is CONSISTENTLY wrong,
so these data-free tests run first. The final sections run only if the CSVs exist.
"""
import os
import pathlib
import sys

import numpy as np
import pandas as pd

import utils
from utils import (CORE_VARIANTS, LANGUAGE_REGION, LANGUAGE_DIC, METHODS_DIC, MODEL_SHORT,
                   GRID_SERIAL, SERIAL_ROLE, REPLICATE_SERIAL_PAIRS,
                   MODEL_FAMILY, TRAIN_EVAL_MATCH, EXCLUDED_MODELS_AGGREGATE,
                   MULTI_CONFIG_TRAIN, WORLDSPEECH_TRAIN_EXAMPLES, TRAIN_CONFIGS,
                   expected_stream_examples, CONFIG_DURATION_AT_CAP, MAX_INPUT_LENGTH_S,
                   KNOWN_AT_CAP_CONFIGS, RESOURCE_TIER, TIER_ORDER, get_eval_domain,
                   get_accent_match, load_tinyaya_composition,
                   add_language_columns, assert_unique_keys, half_up,
                   is_excluded_from_aggregate)

ok = True


def check(name, cond):
    global ok
    ok = ok and bool(cond)
    print(f'[{"PASS" if cond else "FAIL"}] {name}')


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


HISTORY_CSV = os.path.join('data', 'raw_serials', 'history_serial_0.csv')
T1_CSV = os.path.join('results_all', 'acc', 't1_sample_efficiency.csv')

# --- 1. Language / region mapping ---------------------------------------------------
# A mis-regioned language is the defect a numeric checker structurally cannot catch: it
# would regenerate every table wrongly and pass every value comparison.
print('\n--- 1. language and region mapping ---')

EXPECTED_LANGUAGES = {'crs_sc', 'en_us', 'es_419', 'fr_fr', 'ha_ng', 'hi_in', 'id_id',
                      'mr_in', 'sw_ke', 'ta_in', 'am_et', 'ur_pk'}

check('LANGUAGE_REGION covers exactly the 12 executed languages',
      set(LANGUAGE_REGION) == EXPECTED_LANGUAGES)
check('LANGUAGE_DIC covers the same 12 languages',
      set(LANGUAGE_DIC) == EXPECTED_LANGUAGES)
check('crs_sc has region None (OOD probe, NOT earth)',
      LANGUAGE_REGION['crs_sc'] is None)
check('exactly one language is region-None',
      sum(v is None for v in LANGUAGE_REGION.values()) == 1)
check('every non-crs language has a region in {earth, fire, water}',
      all(v in ('earth', 'fire', 'water')
          for k, v in LANGUAGE_REGION.items() if k != 'crs_sc'))
check('earth = {ha_ng, sw_ke, am_et}',
      {k for k, v in LANGUAGE_REGION.items() if v == 'earth'} == {'ha_ng', 'sw_ke', 'am_et'})
check('fire = {hi_in, mr_in, ta_in, ur_pk}',
      {k for k, v in LANGUAGE_REGION.items() if v == 'fire'}
      == {'hi_in', 'mr_in', 'ta_in', 'ur_pk'})
check('water = {en_us, es_419, fr_fr, id_id}',
      {k for k, v in LANGUAGE_REGION.items() if v == 'water'}
      == {'en_us', 'es_419', 'fr_fr', 'id_id'})
check('every region has >=2 languages, so a matched/mismatched contrast exists',
      all(sum(1 for v in LANGUAGE_REGION.values() if v == r) >= 2
          for r in ('earth', 'fire', 'water')))

# The non-clean cells must stay flagged: a region claim leaning on them is weaker than it
# looks, and nothing in the numbers reveals that.
check('fr_fr is flagged as dialect_mismatch (trains fr_ca, evaluates fr_fr)',
      TRAIN_EVAL_MATCH.get('fr_fr') == 'dialect_mismatch')
# The Spain-Spanish runs were deleted and re-run from es_mx, which is inside the Latin
# American variety FLEURS es_419 evaluates -- so it is no longer a mismatch.
check('es_419 is NOT flagged as dialect_mismatch any more (trains es_mx now)',
      TRAIN_EVAL_MATCH.get('es_419') is None)
check('es_419 trains on es_mx, not es_es',
      TRAIN_CONFIGS['es_419'][1] == ('es_mx',))
# Interleaving is NOT a confound and must not be re-added as one: the loader uses
# 'all_exhausted_without_replacement', so the combined stream is exactly the sum of its parts.
# verify_interleave_semantics.py proves it; this pins the conclusion in the data dicts.
check('no language is flagged for interleaving (refuted -- see verify_interleave_semantics)',
      not any(v == 'uniform_interleave' for v in TRAIN_EVAL_MATCH.values()))
check('the multi-config languages are recorded, without implying oversampling',
      set(MULTI_CONFIG_TRAIN) == {'ha_ng', 'sw_ke', 'ur_pk'})
check('every multi-config language lists exactly 2 configs',
      all(len(v) == 2 for v in MULTI_CONFIG_TRAIN.values()))

# The example-count snapshot replaced an hours snapshot taken from a summarised reading of
# the WorldSpeech paper. That was the weakest input in the analysis and it produced a false
# data-integrity finding, so these pin the replacement's shape.
check('every training config in TRAIN_CONFIGS has a known example count',
      all(all(c in WORLDSPEECH_TRAIN_EXAMPLES for c in cfgs)
          for lang, (path, cfgs, split) in TRAIN_CONFIGS.items()
          if path == 'disco-eth/WorldSpeech'))
check('TRAIN_CONFIGS covers all 10 languages', set(TRAIN_CONFIGS) == EXPECTED_LANGUAGES)
check('the multi-config languages in TRAIN_CONFIGS match MULTI_CONFIG_TRAIN',
      {l for l, (_, c, _) in TRAIN_CONFIGS.items() if len(c) > 1}
      == set(MULTI_CONFIG_TRAIN))
check('expected_stream_examples sums the parts for a multi-config language',
      expected_stream_examples('sw_ke')
      == sum(WORLDSPEECH_TRAIN_EXAMPLES[c] for c in MULTI_CONFIG_TRAIN['sw_ke']))
check('expected_stream_examples is None for crs_sc (ERISLab mirror not in snapshot)',
      expected_stream_examples('crs_sc') is None)
check('every example count is a positive int',
      all(isinstance(v, int) and v > 0 for v in WORLDSPEECH_TRAIN_EXAMPLES.values()))

# --- 1b. The duration cap and its silent data loss ----------------------------------
# The defect that produced this study's central finding, pinned so it cannot regress quietly.
print('\n--- 1b. duration cap ---')


def _upstream_filter(length, min_input_length=None, max_input_length=MAX_INPUT_LENGTH_S):
    """The upstream comparison, reproduced. Strict `<`, which is the whole problem."""
    if min_input_length and max_input_length:
        return length > min_input_length and length < max_input_length
    if min_input_length:
        return length > min_input_length
    return length < max_input_length


check('MAX_INPUT_LENGTH_S is 30', MAX_INPUT_LENGTH_S == 30)
check('a 29.99 s clip is KEPT', _upstream_filter(29.99))
check('a clip at exactly the cap is DROPPED -- the bug', not _upstream_filter(30.0))
check('a 30.01 s clip is dropped', not _upstream_filter(30.01))

check('ta_in is recorded as 0% at the cap (proven by the filter test, not sampled)',
      CONFIG_DURATION_AT_CAP['ta_in'] == 0.00)
check('fr_ca is recorded as a partial loss',
      0 < CONFIG_DURATION_AT_CAP['fr_ca'] < 0.5)
check('every at-cap fraction is a proportion',
      all(0.0 <= v <= 1.0 for v in CONFIG_DURATION_AT_CAP.values()))
check('no config is entirely at the cap, so none is excluded from the screen',
      tuple(KNOWN_AT_CAP_CONFIGS) == ())
check('every known-at-cap config is actually at/above the threshold in the snapshot',
      all(CONFIG_DURATION_AT_CAP.get(c, 0) >= 0.5 for c in KNOWN_AT_CAP_CONFIGS))

# Tamil trains on one config, so pre- and post-filter agree.
check('ta_in stream is 8846 pre- and post-filter',
      expected_stream_examples('ta_in') == 8846
      and expected_stream_examples('ta_in', post_filter=True) == 8846)
check('Tamil is a single-config stream',
      TRAIN_CONFIGS['ta_in'][1] == ('ta_in',) and 'ta_in' not in MULTI_CONFIG_TRAIN)
check('post_filter never exceeds pre-filter for any language',
      all(expected_stream_examples(l, post_filter=True) <= expected_stream_examples(l)
          for l in TRAIN_CONFIGS if expected_stream_examples(l) is not None))
check('languages with no at-cap loss are unchanged by post_filter',
      all(expected_stream_examples(l, post_filter=True) == expected_stream_examples(l)
          for l in ('hi_in', 'sw_ke', 'ha_ng', 'id_id', 'mr_in', 'en_us')))

# ta_in must NOT be excluded: the loss hit every decoder arm equally, so the within-language
# contrast is unbiased, and dropping the language would discard the only low-resource cell.
check('ta_in is NOT excluded from aggregates for any core variant',
      not any(is_excluded_from_aggregate(m, 'ta_in') for m in METHODS_DIC))

# --- 1c. resource tiers, eval axes, and the Tiny Aya composition --------------------
print('\n--- 1c. tiers, eval axes, composition ---')
check('RESOURCE_TIER covers all 12 languages', set(RESOURCE_TIER) == EXPECTED_LANGUAGES)
check('every tier label is known',
      set(RESOURCE_TIER.values()) <= set(TIER_ORDER))
check('all four tiers are populated (that was the point of adding am_et and ur_pk)',
      set(RESOURCE_TIER.values()) == set(TIER_ORDER))
check('am_et and ta_in are the very-low tier (~40 h)',
      {k for k, v in RESOURCE_TIER.items() if v == 'very_low'} == {'am_et', 'ta_in'})
check('ur_pk is the low tier (~80 h)',
      {k for k, v in RESOURCE_TIER.items() if v == 'low'} == {'ur_pk'})
check('ha_ng and mr_in are the mid tier (~110-120 h)',
      {k for k, v in RESOURCE_TIER.items() if v == 'mid'} == {'ha_ng', 'mr_in'})

check('only ha_ng and crs_sc evaluate in-domain (WorldSpeech); the rest are FLEURS',
      {l for l in EXPECTED_LANGUAGES if get_eval_domain(l) == 'in_domain'}
      == {'ha_ng', 'crs_sc'})
check('fr_fr is the only different-accent cell', get_accent_match('fr_fr') == 'different')
check('es_419 is a related-accent cell, not different', get_accent_match('es_419') == 'related')
check('en_us is a same-accent cell', get_accent_match('en_us') == 'same')

_comp = load_tinyaya_composition()
if _comp is None:
    print('[SKIP] tinyaya composition CSV absent -- run fetch_tinyaya_composition.py')
else:
    check('composition covers 11 of the 12 languages (crs_sc is not a Tiny Aya language)',
          set(_comp['dataset']) == EXPECTED_LANGUAGES - {'crs_sc'})
    check('every variant column is present',
          {'earth', 'fire', 'global', 'water'} <= set(_comp.columns))
    # The exposure numbers must reproduce the regional design, or the join is wrong.
    check('African languages have their largest exposure in the earth mix',
          all(_comp.set_index('dataset').loc[l, 'earth']
              >= max(_comp.set_index('dataset').loc[l, v] for v in ('fire', 'water'))
              for l in ('am_et', 'ha_ng', 'sw_ke')))
    check('South Asian languages have their largest exposure in the fire mix',
          all(_comp.set_index('dataset').loc[l, 'fire']
              > max(_comp.set_index('dataset').loc[l, v] for v in ('earth', 'water'))
              for l in ('hi_in', 'mr_in', 'ta_in', 'ur_pk')))

# --- 2. Model dicts -----------------------------------------------------------------
print('\n--- 2. model dicts ---')

check('METHODS_DIC keys are the real logged model_ids (whisper-medium, full HF paths)',
      all(k.startswith('q2a_openai/whisper-medium_') for k in METHODS_DIC))
check('no seeded whisper_small placeholder keys survive',
      not any('whisper_small' in k for k in METHODS_DIC))
check('METHODS_DIC, MODEL_SHORT, MODEL_FAMILY share one key set',
      set(METHODS_DIC) == set(MODEL_SHORT) == set(MODEL_FAMILY))
check('CORE_VARIANTS is the 4 grid-wide variants only',
      set(CORE_VARIANTS) == {'earth', 'fire', 'global', 'water'})
check('base and qwen3-4b are NOT in CORE_VARIANTS (they exist only for crs_sc)',
      'base' not in CORE_VARIANTS and 'qwen3-4b' not in CORE_VARIANTS)
check('Qwen3-4B is labelled a non-Aya control, not a TinyAya variant',
      MODEL_FAMILY['q2a_openai/whisper-medium_Qwen/Qwen3-4B'] == 'Non-Aya control')
check('MODEL_SHORT values are unique (a duplicate would merge two variants)',
      len(set(MODEL_SHORT.values())) == len(MODEL_SHORT))

# --- 3. Exclusions ------------------------------------------------------------------
print('\n--- 3. aggregate exclusions ---')

# en_us/water was excluded as a "failed run" until it was re-run and REPLICATED (best 12.05
# vs 17.06, both far worse than the other variants). Two bad runs is a real effect, not a
# failure -- and since water is English's MATCHED variant, excluding it would have removed
# the grid's strongest against-hypothesis point.
check('no runs are excluded from aggregates any more',
      len(EXCLUDED_MODELS_AGGREGATE) == 0)
check('en_us / water is NOT excluded (it replicates)',
      not is_excluded_from_aggregate(
          'q2a_openai/whisper-medium_CohereLabs/tiny-aya-water', 'en_us'))
check('every exclusion, if any is ever added, carries a reason',
      all(e.get('reason') for e in EXCLUDED_MODELS_AGGREGATE))

# --- 4. Rounding and merge guards ---------------------------------------------------
print('\n--- 4. rounding and merge guards ---')

# Python's round() is banker's rounding and disagrees with these on exact ties.
check('half_up(5.25, 1) == 5.3 (round() gives 5.2)', half_up(5.25, 1) == 5.3)
check('half_up(2.5, 0) == 3.0 (round() gives 2)', half_up(2.5, 0) == 3.0)
check('half_up(68.2479, 2) == 68.25', half_up(68.2479, 2) == 68.25)
check('half_up passes None through', half_up(None, 2) is None)
check('half_up handles NaN', half_up(float('nan'), 2) is None)

# The cross-join guard. Keying a merge on a non-unique column turns k rows into k^2 pairs,
# which is invisible in the output -- it just looks like a larger, more reassuring n.
_uniq = pd.DataFrame({'model_id': ['a', 'b'], 'dataset': ['x', 'x'], 'v': [1, 2]})
_dupe = pd.DataFrame({'model_id': ['a', 'a'], 'dataset': ['x', 'x'], 'v': [1, 2]})
check('assert_unique_keys passes on a unique key',
      assert_unique_keys(_uniq, ['model_id', 'dataset']) is not None)
check('assert_unique_keys raises on a duplicated key -- the bug this pins',
      raises(lambda: assert_unique_keys(_dupe, ['model_id', 'dataset']), AssertionError))
check('a language-only key IS duplicated here (k^2 cross-join shape)',
      raises(lambda: assert_unique_keys(_uniq, ['dataset']), AssertionError))

# --- 5. Derived columns -------------------------------------------------------------
print('\n--- 5. derived columns ---')

_df = pd.DataFrame({
    'dataset': ['crs_sc', 'ta_in', 'en_us'],
    'model_id': ['q2a_openai/whisper-medium_CohereLabs/tiny-aya-earth'] * 3,
})
_out = add_language_columns(_df)
check('add_language_columns marks crs_sc ood_encoder_and_decoder',
      _out.loc[0, 'language_status'] == 'ood_encoder_and_decoder')
check('add_language_columns defaults other languages to in_domain',
      _out.loc[2, 'language_status'] == 'in_domain')
check('add_language_columns defaults train_eval_match to clean',
      _out.loc[0, 'train_eval_match'] == 'clean')
# ta_in is 'clean' on this axis now: interleaving is not a confound. Its data-accounting
# anomaly is tracked in t4, not in TRAIN_EVAL_MATCH.
check('add_language_columns no longer flags ta_in for interleaving',
      _out.loc[1, 'train_eval_match'] == 'clean')
_dial = add_language_columns(pd.DataFrame({
    'dataset': ['fr_fr'],
    'model_id': ['q2a_openai/whisper-medium_CohereLabs/tiny-aya-water']}))
check('add_language_columns carries the dialect_mismatch flag',
      _dial.loc[0, 'train_eval_match'] == 'dialect_mismatch')
check('add_language_columns maps model_short', _out.loc[0, 'model_short'] == 'earth')

# --- 6. Data-shape invariants (only if the downloads have run) ----------------------
print('\n--- 6. downloaded-data invariants ---')

if not os.path.exists(HISTORY_CSV):
    print(f'[SKIP] {HISTORY_CSV} not present -- run plotter.sh first')
else:
    h = pd.read_csv(HISTORY_CSV)
    fin = h[h['state'] == 'finished']

    # Effective batch is bs x grad_accum. Reading batch_size alone gives a sample count 64x
    # too small, which turns seconds-of-audio-per-sample into an impossible number.
    check('effective_batch == batch_size * gradient_accumulation_steps == 512',
          set(h['effective_batch'].dropna().unique()) == {512})

    check('every finished run has exactly 202 history rows',
          set(fin.groupby('run_id').size().unique()) == {202})
    check('every finished run has exactly 101 eval rows',
          set(fin[fin['eval/cer'].notna()].groupby('run_id').size().unique()) == {101})
    check('every run has a step-0 row with audio_hours == 0.0',
          (h[h['_step'] == 0]['audio_hours'] == 0.0).all()
          and h[h['_step'] == 0]['run_id'].nunique() == h['run_id'].nunique())
    check('audio_hours is never NaN (the step-0 fill worked)',
          h['audio_hours'].notna().all())
    check('audio_hours is non-decreasing within every run',
          all(g.sort_values('_step')['audio_hours'].is_monotonic_increasing
              for _, g in h.groupby('run_id')))
    check('every logged model_id is in METHODS_DIC',
          set(h['model_id'].dropna().unique()) <= set(METHODS_DIC))
    check('every logged language is in LANGUAGE_REGION',
          set(h['dataset'].dropna().unique()) <= set(LANGUAGE_REGION))
    check('all runs share serial 0', set(h['serial'].dropna().unique()) == {0})

    # Whisper's max_input_length cap is 30 s, so mean seconds per sample must sit under it.
    # This is the check that would have caught the missing gradient-accumulation factor.
    per_run = fin.groupby('run_id').agg(
        secs=('train_audio_seconds_filled', 'max'),
        steps=('train/global_step', 'max'),
        eb=('effective_batch', 'max'))
    sec_per_sample = per_run['secs'] / (per_run['steps'] * per_run['eb'])
    check('mean audio seconds per sample is within the 30 s max_input_length cap',
          bool((sec_per_sample > 0).all() and (sec_per_sample <= 30).all()))

if not os.path.exists(T1_CSV):
    print(f'[SKIP] {T1_CSV} not present -- run plotter.sh first')
else:
    t1 = pd.read_csv(T1_CSV)
    # t1 is a per-RUN table, so run_id is the true key. (model_id, dataset) is unique only
    # among canonical rows -- during a re-run window a cell legitimately holds two runs in the
    # same serial, and every downstream merge filters to the canonical one.
    check('t1 has one row per run_id', not t1.duplicated(subset=['run_id']).any())
    check('t1 has one canonical row per (model_id, dataset)',
          not t1[t1['is_canonical']].duplicated(subset=['model_id', 'dataset']).any())
    check('every (model_id, dataset) cell has exactly one canonical run',
          bool((t1.groupby(['model_id', 'dataset'])['is_canonical'].sum() == 1).all()))
    # Stability rule: an unfinished re-run must never displace the completed run it replaces,
    # or a `bash plotter.sh` mid-re-run would silently rewrite every number from a 7-step curve.
    check('no canonical run is unfinished while a finished run exists in the same cell',
          not any((g['is_canonical'] & (g['state'] != 'finished')).any()
                  and (g['state'] == 'finished').any()
                  for _, g in t1.groupby(['model_id', 'dataset'])))
    check('best_cer <= final_cer for every run (best is a minimum over the curve)',
          bool((t1['best_cer'] <= t1['final_cer'] + 1e-9).all()))
    check('final_minus_best is never negative',
          bool((t1['final_minus_best'] >= -1e-9).all()))
    check('audio_h_to_best <= audio_h_total for every run',
          bool((t1['audio_h_to_best'] <= t1['audio_h_total'] + 1e-9).all()))
    check('threshold hours are ordered: 1.25x >= 1.5x >= 2.0x reach times',
          bool((t1['audio_h_to_1.25x_best'].fillna(np.inf)
                >= t1['audio_h_to_1.5x_best'].fillna(np.inf) - 1e-9).all()
               and (t1['audio_h_to_1.5x_best'].fillna(np.inf)
                    >= t1['audio_h_to_2x_best'].fillna(np.inf) - 1e-9).all()))
    check('no rows are flagged excluded_from_aggregate (en_us/water replicated)',
          int(t1['excluded_from_aggregate'].sum()) == 0)
    # The serial split is the contract: serial 0 is the analysis population and nothing else
    # is in it, so a cross-language aggregate over it is correct without further filtering.
    grid = t1[t1['serial'] == GRID_SERIAL]
    check('serial 0 is exactly 12 languages x 4 variants, 48 runs',
          len(grid) == 48 and grid['dataset'].nunique() == 12
          and set(grid['model_short']) == set(CORE_VARIANTS))
    check('every serial-0 language carries exactly the 4 core variants',
          all(grid[grid['dataset'] == l]['model_id'].nunique() == 4
              for l in grid['dataset'].unique()))
    check('serial 0 has no duplicated (model, language) cell',
          not grid.duplicated(subset=['model_id', 'dataset']).any())
    # The control arms are OUT of the grid but must still reach the OOD table.
    ctrl = t1[t1['serial'] == 2]
    check('serial 2 holds only non-core variants',
          len(ctrl) > 0 and not set(ctrl['model_short']) & set(CORE_VARIANTS))
    check('crs_sc still carries all 6 models across serials 0 and 2',
          t1[t1['dataset'] == 'crs_sc']['model_id'].nunique() == 6)
    check('every serial present in t1 has a declared role',
          set(t1['serial'].unique()) <= set(SERIAL_ROLE))
    # es_es was re-added from another project and shares `dataset: es_419` with es_mx. It is
    # older, and mark_canonical breaks ties by earliest -- so if it were still in serial 0 it
    # would silently become the canonical Spanish run.
    if 'dataset_train' in t1.columns:
        check('no es_es-trained run remains in serial 0',
              not grid['dataset_train'].astype(str).str.contains('es_es').any())

# --- replicate seed classification -------------------------------------------------------
# A pair must land in exactly one of three buckets. The trap this pins: an unrecorded seed
# left as NaN compares unequal to itself, so a naive `!=` calls it "seed varies" and pools a
# nondeterminism pair into the seed-sensitivity estimate, inflating it. The sentinel must not
# be guessed into `same_seed` either -- an unlogged run is not evidence that it used 42.
from analyze_replicates import build_pairs as _build_pairs
from download_wandb_history import UNRECORDED_SEED as _UNRECORDED

def _seed_case(seed_first, seed_rerun):
    def cell(seed, best, label):
        return dict(serial_label=label, run_id=f'r{label}', model_id='m', dataset='en_us',
                    state='finished', seed=seed, best_cer=best, final_cer=best, late_sd=0.4)
    pair = _build_pairs(pd.DataFrame([cell(seed_rerun, 12.0, 'serial_0')]),
                        pd.DataFrame([cell(seed_first, 17.0, 'serial_1')]))
    return pair['seed_status'].iloc[0]

check('two runs at the same seed are same_seed', _seed_case(42, 42) == 'same_seed')
check('42 against 420 is seed_varies', _seed_case(42, 420) == 'seed_varies')
check('the unrecorded sentinel is never read as a real seed',
      _seed_case(_UNRECORDED, 42) == 'unrecorded')
check('a NaN seed is unrecorded, not seed_varies (NaN != NaN trap)',
      _seed_case(np.nan, 42) == 'unrecorded')
check('the sentinel cannot collide with a seed any run actually used',
      _UNRECORDED not in (42, 420))

# --- selection-split reuse ---------------------------------------------------------------
# ha_ng selected its best checkpoint on `disco-eth/WorldSpeech ha_ng test`, the same config the
# eval sweeps use as its in-domain point -- so that number is not held out. Derived from the
# training runs rather than trusting the hand-written dict.
from utils import SELECTION_SPLIT, REUSES_SELECTION_SPLIT, is_selection_split

_hist = os.path.join('data', 'raw_serials', 'history_serial_0.csv')
if not os.path.exists(_hist):
    print(f'[SKIP] {_hist} not present')
else:
    _h = pd.read_csv(_hist).drop_duplicates('dataset')
    _obs = {r.dataset: (r.dataset_path, r.dataset, r.split) for r in _h.itertuples()}
    check('SELECTION_SPLIT matches what the training runs actually evaluated on',
          all(SELECTION_SPLIT.get(k) == v for k, v in _obs.items()))
    check('every study cell has a recorded selection split',
          set(_obs) <= set(SELECTION_SPLIT))
    # Exactly one cell's in-domain eval config collides with its own selection split. If a
    # second ever appears, the substitute table must grow with it.
    _collide = {lang for lang, (dp, ds, sp) in SELECTION_SPLIT.items()
                if dp != 'google/fleurs' and sp in ('test', 'test_clean')}
    check('exactly the cells in REUSES_SELECTION_SPLIT collide with their own eval split',
          _collide == set(REUSES_SELECTION_SPLIT))
    check('ha_ng in-domain eval is flagged as a selection split',
          is_selection_split('ha_ng', 'disco-eth/WorldSpeech', 'ha_ng', 'test'))
    check('crs_sc test_clean is NOT a selection split (it selected on val_clean)',
          not is_selection_split('crs_sc', 'ERISLab/WorldSpeech', 'crs_sc', 'test_clean'))
    check('FLEURS test is NOT a selection split (they selected on validation)',
          not any(is_selection_split(l, 'google/fleurs', l, 'test')
                  for l in ('en_us', 'ta_in', 'ur_pk')))
    check('ha_td is in the Hausa training mix, so the substitute is genuinely in-domain',
          'ha_td' in MULTI_CONFIG_TRAIN['ha_ng'])


# --- in-domain primary -------------------------------------------------------------------
# One WorldSpeech config per cell is its in-domain point; the rest are accent transfer. It is
# the trained variety everywhere except ha_ng, which must use ha_td because its trained variety
# is also its selection split.
from utils import IN_DOMAIN_PRIMARY, in_domain_role, to_study_cell

check('every study cell has an in-domain primary',
      set(IN_DOMAIN_PRIMARY) == set(SELECTION_SPLIT))
check('each primary normalises back to its own cell',
      all(to_study_cell(v) == k for k, v in IN_DOMAIN_PRIMARY.items()))
check('Hausa uses ha_td, not its selection split ha_ng',
      IN_DOMAIN_PRIMARY['ha_ng'] == 'ha_td' == REUSES_SELECTION_SPLIT['ha_ng'])
check('no primary is its own cell selection split',
      not any(is_selection_split(cell, 'disco-eth/WorldSpeech', cfg, 'test')
              for cell, cfg in IN_DOMAIN_PRIMARY.items()))
# Everywhere else the primary IS the trained variety -- ha_ng is the only substitution.
check('ha_ng is the only cell whose primary is not its trained variety',
      {k for k, v in IN_DOMAIN_PRIMARY.items()
       if v != (TRAIN_CONFIGS[k][1][0] if isinstance(TRAIN_CONFIGS[k][1], tuple)
                else TRAIN_CONFIGS[k][1])} == {'ha_ng'})
check('a non-primary WorldSpeech variant is labelled accent_transfer',
      in_domain_role('ha_ng', 'ha_ng', 'in_domain') == 'accent_transfer'
      and in_domain_role('en_us', 'en_au', 'in_domain') == 'accent_transfer')
check('the primary is labelled primary',
      in_domain_role('ha_ng', 'ha_td', 'in_domain') == 'primary'
      and in_domain_role('en_us', 'en_us', 'in_domain') == 'primary')
check('cross-domain rows get no in-domain role',
      in_domain_role('en_us', 'en_us', 'cross_domain') == '')

# Every WorldSpeech variant the sweeps evaluate must map to a study cell, or it carries no
# language_name and no resource_tier and drops out of any grouped table.
import re as _re
_gen = pathlib.Path('for_quantizedasr/tools/preprocess/create_yamls_worldspeech_lalm.py')
if _gen.exists():
    _all = _re.findall(r'"([^"]+)"',
                       _re.search(r'^configs = \[(.*?)^\]', _gen.read_text(), _re.S | _re.M).group(1))
    _study = {c for c in _all if c.split('_')[0] in
              {'en', 'es', 'fr', 'ha', 'sw', 'ur', 'ta', 'hi', 'id', 'mr', 'am', 'crs'}}
    check('every evaluated WorldSpeech variant maps to a study cell',
          all(to_study_cell(c) in SELECTION_SPLIT for c in _study))
    check('exactly one variant per cell is the in-domain primary, the rest accent transfer',
          all(sum(1 for c in _study if to_study_cell(c) == cell
                  and in_domain_role(cell, c, 'in_domain') == 'primary') == 1
              for cell in SELECTION_SPLIT))

# --- serial roles ------------------------------------------------------------------------
check('every replicate pairing names declared serials',
      all(a in SERIAL_ROLE and b in SERIAL_ROLE for a, b in REPLICATE_SERIAL_PAIRS))
check('the grid serial is the canonical side of a replicate pairing',
      any(a == GRID_SERIAL for a, _ in REPLICATE_SERIAL_PAIRS))
check('no serial is both canonical and superseded',
      not {a for a, _ in REPLICATE_SERIAL_PAIRS} & {b for _, b in REPLICATE_SERIAL_PAIRS})

# --- the eval-dataset registry -------------------------------------------------------------
# The dataset list used to be typed in three places and drifted: the baselines sweep had 43
# entries and the trained sweep 44, differing on exactly the config that must not be evaluated.
# There is now one registry and membership is derived, so these check the derivation, not a
# copy-for-copy match.
import missing_runs as _mr
from utils import is_selection_split as _iss

_reg_path = pathlib.Path(_mr.REGISTRY)
if not _reg_path.exists():
    print(f'[SKIP] {_reg_path} not present -- run create_yamls_worldspeech_lalm.py')
else:
    _reg = pd.read_csv(_reg_path)
    check('every registry row resolves to a dataset key',
          all(_mr.dataset_key(c, _reg) is not None for c in _reg['config_yaml']))
    check('use_in_sweep is exactly "not a selection split"',
          all(bool(r['use_in_sweep']) != bool(
              _iss(r['study_cell'], r['dataset_path'], r['dataset'], r['split']))
              for _, r in _reg.iterrows()))
    check('exactly one config is excluded, and it is ha_ng\'s selection split',
          list(_reg[~_reg['use_in_sweep']]['config_yaml'])
          == ['short_ml/worldspeech_ha_ng_test.yaml'])
    check('every study cell has exactly one in-domain primary in the registry',
          (_reg[_reg['in_domain_role'] == 'primary'].groupby('study_cell').size() == 1).all())
    check('Hausa\'s primary is ha_td, not the excluded ha_ng',
          _reg[(_reg['study_cell'] == 'ha_ng')
               & (_reg['in_domain_role'] == 'primary')]['dataset'].tolist() == ['ha_td'])
    check('the registry holds 44 configs, 43 swept',
          len(_reg) == 44 and int(_reg['use_in_sweep'].sum()) == 43)
    check('missing_runs MODEL_CONFIGS excludes the instruct Qwen',
          not any('instruct' in c for c in _mr.MODEL_CONFIGS))


# --- serial 11 checkpoint ids --------------------------------------------------------------
# A trained checkpoint's model_id names the checkpoint, not the model, so serial 11 joins to
# nothing until it is reduced to its parent.
from utils import parent_model_id as _pmi, checkpoint_fields as _cf

_s11 = pathlib.Path('data/raw_serials/raw_serial_11.csv')
check('a parent model_id passes through parent_model_id unchanged',
      _pmi('q2a_openai/whisper-medium_CohereLabs/tiny-aya-fire')
      == 'q2a_openai/whisper-medium_CohereLabs/tiny-aya-fire')
check('a checkpoint id reduces to its METHODS_DIC parent',
      _pmi('ERISLab/q2a_openai_whisper-medium_CohereLabs_tiny-aya-water_ws-ha_ng-700')
      == 'q2a_openai/whisper-medium_CohereLabs/tiny-aya-water')
check('checkpoint_fields recovers the training language and step',
      _cf('ERISLab/q2a_openai_whisper-medium_CohereLabs_tiny-aya-water_ws-ha_ng-700')
      == ('ha_ng', 700))
check('a parent id yields no checkpoint fields',
      _cf('q2a_openai/whisper-medium_CohereLabs/tiny-aya-fire') == (None, None))
if _s11.exists():
    _d = pd.read_csv(_s11)
    check('every serial-11 model_id reduces onto a METHODS_DIC key',
          all(_pmi(m) in METHODS_DIC for m in _d['model_id'].unique()))
    check('every serial-11 checkpoint records a training language',
          all(_cf(m)[0] is not None for m in _d['model_id'].unique()))

# --- held-out vs seen varieties ---------------------------------------------------------
# 'accent_transfer' means "not the cell's primary point", which includes varieties the cell
# TRAINED on. Pooling the two turned a 0/11 transfer result into 2/13.
for _v in ('sw_tz', 'ur_in', 'ha_td'):
    check(f'{_v} is a trained variety, not held-out', utils.is_trained_variety(_v))
for _v in ('en_au', 'es_cl', 'ta_lk', 'es_es'):
    check(f'{_v} is genuinely held out', not utils.is_trained_variety(_v))
check('TRAINED_VARIETIES is derived from TRAIN_CONFIGS, not typed separately',
      utils.TRAINED_VARIETIES
      == frozenset(c for _, cfgs, _ in utils.TRAIN_CONFIGS.values() for c in cfgs))

print(f'\n{"ALL TESTS PASSED" if ok else "SOME TESTS FAILED"}')
sys.exit(0 if ok else 1)
