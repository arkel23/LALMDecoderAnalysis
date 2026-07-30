"""
Reconstructs how much data each training stream consumed, using only what wandb logged plus
authoritative dataset example counts.

Analysis-only by design: nothing here loads a dataset or touches the training framework.
Per run it uses global_step, batch_size, gradient_accumulation_steps, train_audio_seconds and
the epoch counter; the expected stream size comes from utils.WORLDSPEECH_TRAIN_EXAMPLES,
read from the HuggingFace dataset builder metadata.

How the reconstruction works. Under streaming=True the Trainer iterates an IterableDataset;
when the stream is exhausted it restarts and train/epoch advances past 1. So:

    samples_processed      = global_step * batch_size * gradient_accumulation_steps
    implied_stream_samples = samples_processed / epoch

meaningful only once the stream has actually wrapped. Two runs sit at epoch exactly 1.000
(en_us, hi_in), meaning it never wrapped -- for those the figure is a LOWER BOUND, and the
CSV says so via estimate_kind.

The expected stream is the SUM of a language's training configs, which is sound because
interleaving is lossless: 'all_exhausted_without_replacement' never recycles an exhausted
config. That is proved offline in verify_interleave_semantics.py and confirmed on the real
Tamil configs with verify_dataset_durations.py --load.

WHAT THIS TABLE IS NOT. It is not a data-integrity check, and a low ratio here is NOT
evidence of corrupt or missing data. An earlier pass made exactly that mistake: it compared
against hour figures recovered from a summarised reading of the WorldSpeech paper, found
Tamil short, and wrote it up as a data problem. Direct testing refuted that -- the Tamil
configs interleave losslessly and the duration-consistency filter removes zero samples.
Dataset integrity is checked by verify_dataset_durations.py, and only there.

What a ratio away from 1.0 actually indicates is an open question about run bookkeeping: the
epoch counter is the weakest input here, and it is entirely possible for the counter, not the
run, to be the odd part. Treat this table as descriptive.

Usage:
    python analyze_data_accounting.py --input_file data/raw_serials/history_serial_0.csv \
        --output_file results_all/acc/t4_data_accounting.csv
"""
import os
import argparse
import numpy as np
import pandas as pd

from utils import (TRAIN_CONFIGS, WORLDSPEECH_TRAIN_EXAMPLES, MULTI_CONFIG_TRAIN,
                   MODEL_SHORT, LANGUAGE_DIC, expected_stream_examples, assert_unique_keys)


# max_input_length in every configs/train/*ws*.yaml is 30 s, so a mean sample duration above
# this is arithmetically impossible and indicates a dropped factor in the sample count.
MAX_INPUT_LENGTH_S = 30

# Ratio band within which a reconstructed stream is considered to reconcile with the known
# example count. Wide, because the epoch counter is coarse.
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

    out['expected_stream_examples'] = out['dataset'].map(expected_stream_examples)
    out['ratio_implied_to_expected'] = (
        out['implied_stream_samples'] / out['expected_stream_examples'])

    # For a multi-config language, does the reconstruction match ONE config rather than the
    # sum? That distinguishes "the bookkeeping is off" from "fewer configs were consumed",
    # and neither is a data-integrity question.
    def _single_ratio(row):
        entry = TRAIN_CONFIGS.get(row['dataset'])
        if not entry or row['n_train_configs'] < 2 or not np.isfinite(
                row['implied_stream_samples']):
            return np.nan
        counts = [WORLDSPEECH_TRAIN_EXAMPLES.get(c) for c in entry[1]]
        counts = [c for c in counts if c]
        if not counts:
            return np.nan
        # closest single-config match
        ratios = [row['implied_stream_samples'] / c for c in counts]
        return min(ratios, key=lambda r: abs(r - 1.0))

    out['ratio_to_closest_single_config'] = out.apply(_single_ratio, axis=1)

    ratio = out['ratio_implied_to_expected']
    known = out['expected_stream_examples'].notna()
    out['accounting_flag'] = np.select(
        [
            ~known,
            ~out['stream_exhausted'],
            known & (ratio >= RATIO_LOW) & (ratio <= RATIO_HIGH),
        ],
        [
            'expected_size_unknown',
            'stream_never_exhausted_lower_bound_only',
            'reconciles',
        ],
        default='does_not_reconcile_see_docstring',
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
                    expected_stream_examples=('expected_stream_examples', 'first'),
                    ratio_implied_to_expected=('ratio_implied_to_expected', 'mean'),
                    ratio_to_closest_single_config=('ratio_to_closest_single_config', 'mean'),
                    accounting_flag=('accounting_flag', 'first'))
               .sort_values('ratio_implied_to_expected'))
    by_lang.to_csv(args.per_language_file, index=False, float_format=FLOAT_FORMAT)
    print(f'Wrote {len(by_lang)} rows to {args.per_language_file}\n')

    show = ['dataset', 'epochs_logged', 'audio_hours_processed', 'implied_stream_samples',
            'expected_stream_examples', 'ratio_implied_to_expected',
            'ratio_to_closest_single_config', 'accounting_flag']
    print(by_lang[show].round(2).to_string(index=False))

    odd = by_lang[by_lang['accounting_flag'] == 'does_not_reconcile_see_docstring']
    if len(odd):
        print(f'\n{len(odd)} language(s) do not reconcile: {", ".join(odd["dataset"])}. '
              f'This is a bookkeeping observation, NOT a data-integrity finding -- '
              f'integrity is checked only by verify_dataset_durations.py.')
    return 0


if __name__ == '__main__':
    main()
