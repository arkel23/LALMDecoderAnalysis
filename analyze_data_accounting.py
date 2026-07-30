"""
Reconstructs how much data each training stream actually contained, using only what is
logged in wandb plus the frozen WorldSpeech hour snapshot in utils.py.

This is analysis-only by design: nothing here loads a dataset or touches the training
framework. Everything is derived from four logged quantities per run --
train/global_step, batch_size, gradient_accumulation_steps and train/train_audio_seconds --
plus the wandb train/epoch counter.

How the reconstruction works, and why the epoch counter is usable here.

Under streaming=True the Trainer iterates an IterableDataset; when the stream is exhausted
it restarts, and train/epoch advances past 1. So epoch is a count of how many times the
stream was consumed, which makes the implied stream size:

    samples_processed  = global_step * batch_size * gradient_accumulation_steps
    implied_stream_samples = samples_processed / epoch
    implied_stream_hours   = audio_hours_processed / epoch

This is only meaningful when the stream was actually exhausted at least once. Two runs report
epoch exactly 1.000 (en_us, hi_in), which means it never wrapped -- for those the implied
figure is a LOWER BOUND on the stream, not an estimate of it, and the CSV says so.

The interleaving of multi-config languages does NOT distort these numbers. The upstream
loader uses stopping_strategy='all_exhausted_without_replacement', so an exhausted config is
never recycled and the combined stream is exactly the sum of its parts; the uniform
probabilities change arrival order only. verify_interleave_semantics.py proves this.

What the comparison against WorldSpeech can and cannot say is governed by the 'scope' field
of the frozen snapshot -- see utils.WORLDSPEECH_HOURS. Three of the ten languages are
'not_comparable' because the report publishes only a language-level aggregate while training
used one specific config, and three are 'lower_bound' because a second config's hours are
unknown. Only four are direct config-to-config comparisons.

Usage:
    python analyze_data_accounting.py --input_file data/raw_serials/history_serial_0.csv \
        --output_file results_all/acc/t4_data_accounting.csv
"""
import os
import argparse
import numpy as np
import pandas as pd

from utils import (WORLDSPEECH_HOURS, MULTI_CONFIG_TRAIN, MODEL_SHORT, LANGUAGE_DIC,
                   assert_unique_keys)


# max_input_length in every configs/train/*ws*.yaml is 30 s, so a mean sample duration above
# this is arithmetically impossible and indicates a dropped factor in the sample count.
MAX_INPUT_LENGTH_S = 30

# Ratio of implied stream hours to published hours outside which a config warrants
# investigation. Deliberately wide: the published figures carry scope caveats.
RATIO_LOW, RATIO_HIGH = 0.5, 2.0

FLOAT_FORMAT = '%.6f'


def build_table(df):
    rows = []
    for run_id, g in df.groupby('run_id', sort=False):
        g = g.sort_values('_step')
        first = g.iloc[0]
        steps = pd.to_numeric(g['train/global_step'], errors='coerce').max()
        eb = pd.to_numeric(g['effective_batch'], errors='coerce').max()
        secs = pd.to_numeric(g['train_audio_seconds_filled'], errors='coerce').max()
        epoch = pd.to_numeric(g['train/epoch'], errors='coerce').max()

        samples = steps * eb if pd.notna(steps) and pd.notna(eb) else np.nan
        hours = secs / 3600.0 if pd.notna(secs) else np.nan

        lang = first.get('dataset')
        rows.append({
            'run_id': run_id,
            'dataset': lang,
            'language_name': LANGUAGE_DIC.get(lang),
            'model_short': MODEL_SHORT.get(first.get('model_id')),
            'state': first.get('state'),
            'global_step': steps,
            'batch_size': first.get('batch_size'),
            'gradient_accumulation_steps': first.get('gradient_accumulation_steps'),
            'effective_batch': eb,
            'samples_processed': samples,
            'audio_hours_processed': hours,
            'epochs_logged': epoch,
            'mean_sample_seconds': secs / samples if samples else np.nan,
            'train_configs': '+'.join(MULTI_CONFIG_TRAIN.get(lang, (lang,))),
            'n_train_configs': len(MULTI_CONFIG_TRAIN.get(lang, (lang,))),
        })

    out = pd.DataFrame(rows)
    assert_unique_keys(out, ['run_id'], label='t4_data_accounting')
    return out


def add_stream_estimates(out):
    ep = out['epochs_logged']

    # Exhausted at least once <=> epoch strictly above 1. Exactly 1.000 means the stream was
    # never consumed a second time, so the implied size is a floor, not an estimate.
    exhausted = ep > 1.0 + 1e-9
    out['stream_exhausted'] = exhausted
    out['estimate_kind'] = np.where(exhausted, 'estimate', 'lower_bound')

    with np.errstate(divide='ignore', invalid='ignore'):
        out['implied_stream_samples'] = out['samples_processed'] / ep
        out['implied_stream_hours'] = out['audio_hours_processed'] / ep

    ws = out['dataset'].map(lambda d: (WORLDSPEECH_HOURS.get(d) or {}).get('hours'))
    out['worldspeech_hours'] = pd.to_numeric(ws, errors='coerce')
    out['worldspeech_scope'] = out['dataset'].map(
        lambda d: (WORLDSPEECH_HOURS.get(d) or {}).get('scope'))
    out['worldspeech_note'] = out['dataset'].map(
        lambda d: (WORLDSPEECH_HOURS.get(d) or {}).get('note'))

    out['ratio_implied_to_published'] = (
        out['implied_stream_hours'] / out['worldspeech_hours'])

    # Only config-scope rows support a direct comparison. lower_bound rows are still
    # informative in ONE direction: if the implied stream is far BELOW a lower bound, that is
    # a real discrepancy, because the true corpus is larger still.
    comparable = out['worldspeech_scope'].isin(['config', 'lower_bound'])
    ratio = out['ratio_implied_to_published']
    out['accounting_flag'] = np.select(
        [
            ~comparable,
            ~out['stream_exhausted'],
            comparable & (ratio < RATIO_LOW),
            comparable & (ratio > RATIO_HIGH),
        ],
        [
            'not_comparable_published_scope',
            'stream_never_exhausted_lower_bound_only',
            'IMPLIED_STREAM_FAR_BELOW_PUBLISHED',
            'implied_stream_above_published',
        ],
        default='consistent',
    )

    out['mean_sample_seconds_within_cap'] = (
        out['mean_sample_seconds'] <= MAX_INPUT_LENGTH_S)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_file', type=str,
                   default=os.path.join('data', 'raw_serials', 'history_serial_0.csv'))
    p.add_argument('--output_file', type=str,
                   default=os.path.join('results_all', 'acc', 't4_data_accounting.csv'))
    p.add_argument('--per_language_file', type=str,
                   default=os.path.join('results_all', 'acc',
                                        't4_data_accounting_by_language.csv'))
    args = p.parse_args()

    df = pd.read_csv(args.input_file)
    out = add_stream_estimates(build_table(df))

    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    out.to_csv(args.output_file, index=False, float_format=FLOAT_FORMAT)
    print(f'Wrote {len(out)} rows to {args.output_file}')

    # Per-language view: every decoder sees the identical stream, so these collapse.
    fin = out[out['state'] == 'finished']
    by_lang = (fin.groupby('dataset', as_index=False)
               .agg(language_name=('language_name', 'first'),
                    n_runs=('run_id', 'count'),
                    train_configs=('train_configs', 'first'),
                    epochs_logged=('epochs_logged', 'mean'),
                    audio_hours_processed=('audio_hours_processed', 'mean'),
                    mean_sample_seconds=('mean_sample_seconds', 'mean'),
                    implied_stream_hours=('implied_stream_hours', 'mean'),
                    implied_stream_samples=('implied_stream_samples', 'mean'),
                    estimate_kind=('estimate_kind', 'first'),
                    worldspeech_hours=('worldspeech_hours', 'first'),
                    worldspeech_scope=('worldspeech_scope', 'first'),
                    ratio_implied_to_published=('ratio_implied_to_published', 'mean'),
                    accounting_flag=('accounting_flag', 'first'))
               .sort_values('ratio_implied_to_published'))
    by_lang.to_csv(args.per_language_file, index=False, float_format=FLOAT_FORMAT)
    print(f'Wrote {len(by_lang)} rows to {args.per_language_file}\n')

    show = ['dataset', 'epochs_logged', 'audio_hours_processed', 'mean_sample_seconds',
            'implied_stream_hours', 'worldspeech_hours', 'worldspeech_scope',
            'ratio_implied_to_published', 'accounting_flag']
    print(by_lang[show].round(2).to_string(index=False))

    bad = by_lang[by_lang['accounting_flag'] == 'IMPLIED_STREAM_FAR_BELOW_PUBLISHED']
    if len(bad):
        print(f'\n{len(bad)} language(s) need investigation: '
              f'{", ".join(bad["dataset"])}')
    return 0


if __name__ == '__main__':
    main()
