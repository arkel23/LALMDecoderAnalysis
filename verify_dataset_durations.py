"""
Dataset-integrity checker: interleave arithmetic and duration consistency, for any
dataset_path / config list / split.

WHY THIS EXISTS, and what it has already settled.

An earlier pass of this analysis inferred a data problem in the Tamil training stream from a
derived ratio, and proposed two mechanisms: that uniform interleave probabilities oversampled
the smaller config, and that a duration-column inconsistency was silently dropping samples.
Both were wrong, and a direct test on the real data is what showed it. This script is that
test, generalised, so the next such suspicion is checked before it is written down.

VERIFIED 2026-07-30 on disco-eth/WorldSpeech, configs ta_in + ta_lk, split train:

  * len(interleaved) == len(ta_in) + len(ta_lk)          -- interleaving loses nothing
  * the duration-consistency filter removed ZERO samples -- no corrupt or mislabelled clips

So there is no data-integrity issue with the Tamil configs. The interleave semantics are
sound (see also verify_interleave_semantics.py, which proves the same property offline with
synthetic data), and the reported `duration` column agrees with the decoded audio.

Two modes, because the cheap one is usually enough:

  default        Metadata only. Reads num_examples per config from the dataset builder --
                 authoritative, no audio downloaded, instant. Checks that the arithmetic an
                 interleaved stream must satisfy (combined == sum of parts) is what the
                 accounting assumes.
  --load         The full check: loads the splits map-style, computes audio_length_s from the
                 decoded arrays, asserts len(interleaved) == sum of parts on the real objects,
                 and applies the duration-consistency filter, reporting exactly how many
                 samples it removes. This downloads audio and can be tens of GB, so it is
                 opt-in and not run by plotter.sh.

Usage:
    # cheap, authoritative sample counts
    python verify_dataset_durations.py --dataset_path disco-eth/WorldSpeech \\
        --dataset_configs ta_in ta_lk --split train

    # the full duration-consistency + interleave check (downloads audio)
    python verify_dataset_durations.py --dataset_path disco-eth/WorldSpeech \\
        --dataset_configs ta_in ta_lk --split train --load --num_proc 20

Exit code is non-zero if a check fails, so it can gate a pipeline.
"""
import os
import sys
import argparse

import pandas as pd

# Duration-consistency tolerance, in seconds. A clip is kept when the decoded audio length
# and the corpus `duration` column agree to within this. Matches the tolerance used when the
# Seychellois Creole splits were cleaned.
DEFAULT_TOL = 1.0

# The sentinel make_audio_length_fn assigns when a clip cannot be decoded. It is deliberately
# huge so the max_input_length filter removes it -- which means corrupt clips disappear
# without being counted anywhere. Counting them is one of the points of this script.
CORRUPT_SENTINEL = 1e5

FLOAT_FORMAT = '%.6f'

ok = True


def check(name, cond, detail=''):
    global ok
    ok = ok and bool(cond)
    print(f'[{"PASS" if cond else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))


def make_audio_length_fn(audio_column='audio'):
    """Duration from the decoded array, never from a metadata column.

    Duplicated rather than imported: this repo does analysis only and must not depend on the
    training framework. It is a verbatim copy of the upstream function, which is the point --
    a re-implementation would be measuring something else.
    """
    def get_audio_length(example):
        audio = example[audio_column]
        try:
            example['audio_length_s'] = len(audio['array']) / audio['sampling_rate']
        except Exception:
            # if len(audio['array']) cannot be read: missing file/corrupted
            example['audio_length_s'] = CORRUPT_SENTINEL
        return example
    return get_audio_length


def metadata_counts(dataset_path, configs, split):
    """num_examples per config, straight from the builder. No audio downloaded."""
    from datasets import load_dataset_builder

    rows = []
    for cfg in configs:
        try:
            splits = load_dataset_builder(dataset_path, cfg).info.splits or {}
            info = splits.get(split)
            rows.append({
                'dataset_path': dataset_path, 'dataset_config': cfg, 'split': split,
                'n_examples': info.num_examples if info else None,
                'num_bytes': info.num_bytes if info else None,
                'available_splits': ','.join(sorted(splits)),
                'error': None,
            })
        except Exception as exc:
            rows.append({
                'dataset_path': dataset_path, 'dataset_config': cfg, 'split': split,
                'n_examples': None, 'num_bytes': None, 'available_splits': None,
                'error': f'{type(exc).__name__}: {exc}',
            })
    return pd.DataFrame(rows)


def full_check(dataset_path, configs, split, num_proc, tol, audio_col):
    """Load for real, assert the interleave arithmetic, and measure duration consistency."""
    from datasets import load_dataset, interleave_datasets

    add_audio_length = make_audio_length_fn(audio_col)

    parts, rows = [], []
    for cfg in configs:
        ds = load_dataset(dataset_path, cfg, split=split, streaming=False)
        ds = ds.map(add_audio_length, num_proc=num_proc)
        parts.append(ds)

        lengths = ds['audio_length_s']
        n_corrupt = sum(1 for x in lengths if x >= CORRUPT_SENTINEL)
        real = [x for x in lengths if x < CORRUPT_SENTINEL]
        rows.append({
            'dataset_path': dataset_path, 'dataset_config': cfg, 'split': split,
            'n_examples': len(ds),
            'audio_hours_computed': sum(real) / 3600.0,
            'mean_audio_seconds': (sum(real) / len(real)) if real else None,
            'n_corrupt_sentinel': n_corrupt,
            'n_ge_30s': sum(1 for x in real if x >= 30),
        })
        check(f'{cfg}: no undecodable clips', n_corrupt == 0, f'{n_corrupt} corrupt')

    per_config = pd.DataFrame(rows)

    # The interleave arithmetic, on the real objects. This is the assertion whose failure the
    # earlier analysis wrongly assumed: 'all_exhausted_without_replacement' never recycles an
    # exhausted config, so the combined stream is exactly the sum of its parts.
    combined = None
    if len(parts) > 1:
        combined = interleave_datasets(
            parts, probabilities=[1 / len(parts)] * len(parts),
            stopping_strategy='all_exhausted_without_replacement')
        expected = sum(len(p) for p in parts)
        check('interleaved length == sum of parts (no loss, no oversampling)',
              len(combined) == expected, f'{len(combined)} vs {expected}')
    else:
        combined = parts[0]

    # Duration consistency. Skipped rather than faked when the corpus has no duration column.
    if 'duration' not in combined.column_names:
        print(f'[SKIP] no `duration` column in {dataset_path} -- '
              f'consistency check not applicable')
        n_removed = None
    else:
        def is_length_consistent(example, tol=tol):
            a, d = example['audio_length_s'], example['duration']
            return a > 0 and abs(a - d) < tol

        filtered = combined.filter(is_length_consistent, num_proc=num_proc)
        n_removed = len(combined) - len(filtered)
        check(f'duration column agrees with decoded audio within {tol}s '
              f'for every clip',
              n_removed == 0,
              f'{n_removed} of {len(combined)} would be removed')

    summary = pd.DataFrame([{
        'dataset_path': dataset_path,
        'dataset_configs': '+'.join(configs),
        'split': split,
        'n_configs': len(configs),
        'n_examples_sum_of_parts': int(sum(len(p) for p in parts)),
        'n_examples_interleaved': int(len(combined)),
        'interleave_lossless': bool(len(combined) == sum(len(p) for p in parts)),
        'audio_hours_computed': float(per_config['audio_hours_computed'].sum()),
        'n_corrupt_sentinel': int(per_config['n_corrupt_sentinel'].sum()),
        'duration_tol_s': tol,
        'n_removed_by_duration_filter': n_removed,
        'duration_consistent': (None if n_removed is None else bool(n_removed == 0)),
    }])
    return per_config, summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset_path', type=str, default='disco-eth/WorldSpeech')
    p.add_argument('--dataset_configs', nargs='+', type=str, required=True,
                   help='One or more configs. Several means the interleave check runs.')
    p.add_argument('--split', type=str, default='train')
    p.add_argument('--load', action='store_true',
                   help='Full check: downloads audio, computes durations, runs the '
                        'interleave and duration-consistency assertions.')
    p.add_argument('--num_proc', type=int, default=20)
    p.add_argument('--tol', type=float, default=DEFAULT_TOL)
    p.add_argument('--audio_col_name', type=str, default='audio')
    p.add_argument('--output_file', type=str, default=None,
                   help='Where to write the summary CSV. Defaults to a name derived from '
                        'the dataset path, configs and split under data/dataset_checks/.')
    return p.parse_args()


def main():
    args = parse_args()

    slug = (f"{args.dataset_path.replace('/', '_')}_"
            f"{'-'.join(args.dataset_configs)}_{args.split}")
    out_file = args.output_file or os.path.join('data', 'dataset_checks', f'{slug}.csv')
    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)

    print(f'\n{args.dataset_path} configs={args.dataset_configs} split={args.split}\n')

    meta = metadata_counts(args.dataset_path, args.dataset_configs, args.split)
    print(meta[['dataset_config', 'n_examples', 'error']].to_string(index=False))

    check('every config resolved', meta['error'].isna().all(),
          '; '.join(x for x in meta['error'].dropna()))
    check(f'every config has a `{args.split}` split',
          meta['n_examples'].notna().all())

    total = meta['n_examples'].sum()
    print(f'\nsum of parts (metadata): {int(total)} examples')

    if args.load:
        per_config, summary = full_check(
            args.dataset_path, args.dataset_configs, args.split,
            args.num_proc, args.tol, args.audio_col_name)
        # Cross-check the loaded counts against the builder metadata -- if these disagree the
        # metadata-only mode cannot be trusted for the accounting.
        merged = meta.merge(per_config, on=['dataset_path', 'dataset_config', 'split'],
                            suffixes=('_meta', '_loaded'))
        agree = (merged['n_examples_meta'] == merged['n_examples_loaded']).all()
        check('loaded example counts match the builder metadata', bool(agree))
        out = summary
        print()
        print(summary.T.to_string())
    else:
        out = pd.DataFrame([{
            'dataset_path': args.dataset_path,
            'dataset_configs': '+'.join(args.dataset_configs),
            'split': args.split,
            'n_configs': len(args.dataset_configs),
            'n_examples_sum_of_parts': int(total) if pd.notna(total) else None,
            'mode': 'metadata_only',
        }])
        print('\n(metadata-only mode: pass --load to download audio and run the '
              'duration-consistency and interleave assertions)')

    out.to_csv(out_file, index=False, float_format=FLOAT_FORMAT)
    meta.to_csv(out_file.replace('.csv', '_per_config.csv'), index=False,
                float_format=FLOAT_FORMAT)
    print(f'\nWrote {out_file}')

    print(f'\n{"ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
