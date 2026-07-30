"""
Logic tests for utils.py and the analysis scripts.

Deliberately not pytest: it is not installed in this repo's conda env, and adding a
dependency for one file would break the "a bare checkout runs `bash plotter.sh`" bar.

The division of labour with verify_paper_numbers.py matters. That script asks "does the
document match the CSVs?". These ask "is the logic right?". A number-checker cannot catch a
CSV that is CONSISTENTLY wrong -- a mis-regioned language would regenerate every table
wrongly and still pass every numeric check -- so the cheap, data-free logic tests run first.

Sections 1-5 are pure logic and need no data. Section 6 runs only if the downloaded CSVs
exist, so this file is still useful on a fresh checkout before plotter.sh has run.
"""
import os
import sys

import numpy as np
import pandas as pd

import utils
from utils import (CORE_VARIANTS, LANGUAGE_REGION, LANGUAGE_DIC, METHODS_DIC, MODEL_SHORT,
                   MODEL_FAMILY, TRAIN_EVAL_MATCH, EXCLUDED_MODELS_AGGREGATE,
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

EXPECTED_LANGUAGES = {'crs_sc', 'en_us', 'es_419', 'fr_fr', 'ha_ng',
                      'hi_in', 'id_id', 'mr_in', 'sw_ke', 'ta_in'}

check('LANGUAGE_REGION covers exactly the 10 executed languages',
      set(LANGUAGE_REGION) == EXPECTED_LANGUAGES)
check('LANGUAGE_DIC covers the same 10 languages',
      set(LANGUAGE_DIC) == EXPECTED_LANGUAGES)
check('crs_sc has region None (OOD probe, NOT earth)',
      LANGUAGE_REGION['crs_sc'] is None)
check('exactly one language is region-None',
      sum(v is None for v in LANGUAGE_REGION.values()) == 1)
check('every non-crs language has a region in {earth, fire, water}',
      all(v in ('earth', 'fire', 'water')
          for k, v in LANGUAGE_REGION.items() if k != 'crs_sc'))
check('earth = {ha_ng, sw_ke}',
      {k for k, v in LANGUAGE_REGION.items() if v == 'earth'} == {'ha_ng', 'sw_ke'})
check('fire = {hi_in, mr_in, ta_in}',
      {k for k, v in LANGUAGE_REGION.items() if v == 'fire'} == {'hi_in', 'mr_in', 'ta_in'})
check('water = {en_us, es_419, fr_fr, id_id}',
      {k for k, v in LANGUAGE_REGION.items() if v == 'water'}
      == {'en_us', 'es_419', 'fr_fr', 'id_id'})
check('every region has >=2 languages, so a matched/mismatched contrast exists',
      all(sum(1 for v in LANGUAGE_REGION.values() if v == r) >= 2
          for r in ('earth', 'fire', 'water')))

# The non-clean cells must stay flagged: a region claim leaning on them is weaker than it
# looks, and nothing in the numbers reveals that.
check('fr_fr and es_419 flagged as dialect_mismatch',
      TRAIN_EVAL_MATCH.get('fr_fr') == 'dialect_mismatch'
      and TRAIN_EVAL_MATCH.get('es_419') == 'dialect_mismatch')
check('ta_in, ha_ng, sw_ke flagged as uniform_interleave',
      all(TRAIN_EVAL_MATCH.get(k) == 'uniform_interleave'
          for k in ('ta_in', 'ha_ng', 'sw_ke')))

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

check('exactly one excluded run is recorded',
      len(EXCLUDED_MODELS_AGGREGATE) == 1)
check('the excluded run is en_us / water',
      is_excluded_from_aggregate(
          'q2a_openai/whisper-medium_CohereLabs/tiny-aya-water', 'en_us'))
check('exclusion is (model, language) specific, not model-wide',
      not is_excluded_from_aggregate(
          'q2a_openai/whisper-medium_CohereLabs/tiny-aya-water', 'fr_fr'))
check('every exclusion carries a reason',
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
check('add_language_columns carries the uniform_interleave flag',
      _out.loc[1, 'train_eval_match'] == 'uniform_interleave')
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
    check('t1 has one row per (model_id, dataset)',
          not t1.duplicated(subset=['model_id', 'dataset']).any())
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
    check('exactly one row is flagged excluded_from_aggregate',
          int(t1['excluded_from_aggregate'].sum()) == 1)
    check('crs_sc is the only language with 6 models',
          t1[t1['dataset'] == 'crs_sc']['model_id'].nunique() == 6)

print(f'\n{"ALL TESTS PASSED" if ok else "SOME TESTS FAILED"}')
sys.exit(0 if ok else 1)
