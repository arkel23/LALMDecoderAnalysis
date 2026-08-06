"""Removes `ta_lk` from the Tamil runs' training-config fields in W&B.

The corpus is pre-segmented into fixed 30 s windows and the upstream filter keeps a clip only
when `length < max_input_length`, strictly. Verified over the FULL config, not a sample: all
23,261 `ta_lk` clips carry `duration` exactly 30.000000 -- a single unique value -- so none
survives the filter and the Tamil stream is `ta_in` alone. The config therefore records a corpus
the model never saw a single example of, which makes every downstream reader wrong about what
Tamil trained on.

Drops `ta_lk` from `dataset_train`, and the entry at the same index from `dataset_path_train`
and `split_train`, so the three stay aligned.

MUTATES a live W&B project, so it is a DRY RUN by default. Pass --execute to apply.

    python purge_ta_lk_from_configs.py
    python purge_ta_lk_from_configs.py --execute
"""
import argparse

import wandb

DROP = 'ta_lk'
PROJECT = 'LisTAya/LALMDecoder'
ALIGNED_FIELDS = ('dataset_path_train', 'dataset_train', 'split_train')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project_name', type=str, default=PROJECT)
    p.add_argument('--drop', type=str, default=DROP)
    p.add_argument('--execute', action='store_true',
                   help='Actually write to W&B. Without it this is a dry run.')
    args = p.parse_args()

    api = wandb.Api()
    changed = 0
    for run in api.runs(path=args.project_name, per_page=2000):
        names = run.config.get('dataset_train')
        if not isinstance(names, list) or args.drop not in names:
            continue

        keep = [i for i, n in enumerate(names) if n != args.drop]
        new = {}
        for field in ALIGNED_FIELDS:
            values = run.config.get(field)
            # Only reindex a field that is aligned with dataset_train; anything else is left
            # alone rather than guessed at.
            if isinstance(values, list) and len(values) == len(names):
                new[field] = [values[i] for i in keep]

        print(f'[{"APPLY" if args.execute else "DRY-RUN"}] {run.id}  serial='
              f'{run.config.get("serial")}  {run.state}')
        for field in ALIGNED_FIELDS:
            if field in new:
                print(f'    {field:19s}: {run.config.get(field)}  ->  {new[field]}')
            else:
                print(f'    {field:19s}: NOT ALIGNED with dataset_train, left unchanged')

        if args.execute:
            for field, values in new.items():
                run.config[field] = values
            run.update()
        changed += 1

    if not changed:
        print(f'No run carries {args.drop!r} in dataset_train -- nothing to do.')
    elif args.execute:
        print(f'\nApplied: {changed} run(s) updated.')
    else:
        print(f'\nDRY RUN over {changed} run(s). Nothing was written. Pass --execute to apply.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
