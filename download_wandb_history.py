"""Downloads the per-step wandb history for every run in a serial, one row per (run, step).

Companion to download_save_wandb_data.py, which pulls one row per run from run.summary -- the
right shape for eval-only runs, the wrong shape for training runs that log a curve. The
argparse / results_dir / sort_save_df skeleton is kept identical to that script.

Usage (one serial per invocation):
    python download_wandb_history.py --project_name LisTAya/LALMDecoder --serials 0 \
        --history --output_file raw_serials/history_serial_0.csv --results_dir data
"""
import os
import argparse
import wandb
import pandas as pd


# Config fields copied onto every step-row so the CSV is self-contained. batch_size and
# gradient_accumulation_steps travel together because the effective batch is their product
# (8 x 64 = 512); reading batch_size alone gives a sample count 64x too small.
CONFIG_COLS = [
    'serial', 'dataset_path', 'dataset', 'split', 'model_id', 'force_asr_language',
    'batch_size', 'gradient_accumulation_steps', 'max_steps', 'lr', 'seed',
    'freeze_encoder', 'freeze_decoder', 'peft',
    'dataset_path_train', 'dataset_train', 'split_train',
    'model/num_parameters', 'max_input_length', 'streaming',
]

# The logged metric keys. These are HF Trainer names ('eval/cer', not 'cer') -- the
# summary downloader's SUMMARY_COLS list does not overlap with these at all.
HISTORY_KEYS = [
    '_step', '_runtime', '_timestamp',
    'train/global_step', 'train/epoch',
    'train/train_audio_seconds', 'train/num_input_tokens_seen',
    'train/loss', 'train/grad_norm', 'train/learning_rate',
    'train/train_tokens_per_second',
    'eval/loss', 'eval/cer', 'eval/runtime',
    'eval/samples_per_second', 'eval/steps_per_second',
]

# An unrecorded seed becomes a sentinel that cannot collide with a real one, paired with
# `seed_recorded`. NaN would be poison: NaN != NaN, so every unrecorded pair would read as
# "seed varies". Guessing 42 would be worse -- an unlogged run is not evidence of 42.
UNRECORDED_SEED = 0

SORT_COLS = ['serial', 'dataset', 'model_id', '_step']

# 6 dp so one rounding is enough: 2 dp storage turned a true 68.2479 into a wrong 68.3.
FLOAT_FORMAT = '%.6f'


def get_wandb_project_runs(project, serials=None):
    api = wandb.Api()

    if serials:
        runs = api.runs(path=project, per_page=2000,
                        filters={'$or': [{'config.serial': s} for s in serials]})
    else:
        runs = api.runs(path=project, per_page=2000)

    print('Downloaded runs: ', len(runs))
    return runs


def make_history_df(runs, config_cols, history_keys):
    frames = []

    for i, run in enumerate(runs):
        rows = list(run.scan_history())
        if not rows:
            print(f'  no history for {run.name}, skipping')
            continue

        df = pd.DataFrame(rows)

        # Ask for every key we know about; absent ones become NaN columns rather than
        # KeyErrors, so one run logging a metric the others do not cannot break the concat.
        for key in history_keys:
            if key not in df.columns:
                df[key] = pd.NA
        df = df[history_keys]

        try:
            host = run.metadata.get('host')
        except Exception:
            host = None

        df.insert(0, 'run_id', run.id)
        df.insert(1, 'run_name', run.name)
        df['state'] = run.state
        df['host'] = host
        for col in config_cols:
            df[col] = _scalarize(run.config.get(col, None))

        if 'seed' in config_cols:
            raw_seed = run.config.get('seed', None)
            df['seed_recorded'] = raw_seed is not None
            if raw_seed is None:
                df['seed'] = UNRECORDED_SEED

        frames.append(df)

        if (i + 1) % 10 == 0:
            print(f'{i + 1}/{len(runs)}')

    if not frames:
        raise SystemExit('No history rows downloaded -- check the project and serial.')

    df = pd.concat(frames, axis=0, ignore_index=True)
    print(f'Collected {len(df)} step-rows from {len(frames)} runs')
    return df


def _scalarize(value):
    """Lists (dataset_train: ['ta_in', 'ta_lk']) would broadcast against the row index and
    raise, so keep them as their string form -- the multi-config languages need to stay
    visible in the CSV because they are interleaved 50/50 regardless of size."""
    if isinstance(value, (list, tuple)):
        return str(list(value))
    return value


def add_derived_columns(df):
    """audio_hours is the x-axis for every training curve.

    The step-0 row is the eval_on_start evaluation: it has an eval/cer but no
    train/train_audio_seconds, because no batch has been consumed yet. Left as NaN the
    first point of every curve silently disappears; it must be 0.0. Later eval rows do
    carry the cumulative value, so a per-run forward fill covers any other gap without
    inventing data (the metric is monotonically cumulative by construction).
    """
    df = df.sort_values(by=['run_id', '_step'])

    secs = df.groupby('run_id')['train/train_audio_seconds'].ffill()
    # Only the leading NaNs survive the ffill, and those are genuinely "before any audio".
    secs = secs.fillna(0.0)

    df['train_audio_seconds_filled'] = secs
    df['audio_hours'] = secs / 3600.0

    # Effective batch, so downstream analysis never re-derives it from batch_size alone.
    if {'batch_size', 'gradient_accumulation_steps'}.issubset(df.columns):
        df['effective_batch'] = (
            pd.to_numeric(df['batch_size'], errors='coerce')
            * pd.to_numeric(df['gradient_accumulation_steps'], errors='coerce').fillna(1)
        )

    df['is_eval_row'] = df['eval/cer'].notna()
    df['is_train_row'] = df['train/loss'].notna()
    return df


def sort_save_df(df, fp, sort_cols=None):
    sort_cols = [c for c in (sort_cols or ['serial']) if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=[True for _ in sort_cols])
    df.to_csv(fp, header=True, index=False, float_format=FLOAT_FORMAT)
    return 0


def parse_args():
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument('--project_name', type=str, default='LisTAya/LALMDecoder',
                        help='project_entity/project_name')
    # Filters
    parser.add_argument('--serials', nargs='+', type=int, default=[0])
    parser.add_argument('--config_cols', nargs='+', type=str, default=CONFIG_COLS)
    parser.add_argument('--history_keys', nargs='+', type=str, default=HISTORY_KEYS)

    # The flag that switches this from a no-op into the time-series download. It exists so
    # plotter.sh can carry a single, explicit "also pull the curves" switch rather than
    # having the behaviour depend on which script name is invoked.
    parser.add_argument('--history', action='store_true',
                        help='Download per-step history. Without it nothing is written.')

    # Output
    parser.add_argument('--output_file', default='raw_serials/history_serial_0.csv',
                        type=str, help='File path, relative to --results_dir')
    parser.add_argument('--results_dir', type=str, default='data',
                        help='The directory where results will be stored')
    parser.add_argument('--sort_cols', nargs='+', type=str, default=SORT_COLS)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if not args.history:
        print('--history not set: nothing to do. '
              'Pass --history to download the per-step time series.')
        return 0

    args.output_file = os.path.join(args.results_dir, args.output_file)
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)

    runs = get_wandb_project_runs(args.project_name, args.serials)
    df = make_history_df(runs, args.config_cols, args.history_keys)
    df = add_derived_columns(df)

    sort_save_df(df, args.output_file, args.sort_cols)
    print(f'Saved {len(df)} rows to {args.output_file}')

    return 0


if __name__ == '__main__':
    main()
