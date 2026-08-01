"""
Moves specific W&B runs to a different `serial`, in place, and keeps the run NAME in sync.

Adapted from the sibling repos' `rename_wandb_serial.py`, which selects runs by a substring
condition on a config column. That selector cannot be used here: the case this script exists
for is a re-run, where the old and the new run share *every* config field -- same model_id,
same dataset, same split, same lr. The only thing distinguishing them is the run id. So
selection is by explicit `--run_ids`.

W&B run names bake in a literal trailing "_<serial>" suffix (e.g. "..._train_0"), and that
does not update itself when config.serial changes. Whenever a selected run's name ends with
"_<from_serial>", the suffix is rewritten too, so name and config stay consistent.

Motivating case (2026-08-01): the en_us / tiny-aya-water cell was run twice --
  n4cot5v7  created 2026-07-27, best eval/cer 17.06
  pwnz2zno  created 2026-07-31, best eval/cer 12.05
Both under serial 0, which breaks the one-row-per-(model, language) contract that every
analysis table in this repo asserts. The ORIGINAL (n4cot5v7) moves to serial 1 so serial 0
stays the canonical grid, and the superseded run stays retrievable rather than deleted.

This MUTATES a live W&B project, so it is a DRY RUN by default: it prints every change it
would make and writes nothing. Pass --execute to apply.

    # see what would change
    python rename_wandb_serial.py --run_ids n4cot5v7 --from_serial 0 --to_serial 1

    # apply it
    python rename_wandb_serial.py --run_ids n4cot5v7 --from_serial 0 --to_serial 1 --execute
"""
import argparse

import wandb


def renamed_run_name(name, from_serial, to_serial):
    """Rewrite a trailing '_<from_serial>' suffix. Returns None when there is nothing to do."""
    old_suffix = f'_{from_serial}'
    new_suffix = f'_{to_serial}'
    if name and str(name).endswith(old_suffix):
        return str(name)[:-len(old_suffix)] + new_suffix
    return None


def rename_serial(project, run_ids, from_serial, to_serial, execute):
    api = wandb.Api()
    serial_updated = name_updated = 0

    for run_id in run_ids:
        run = api.run(f'{project}/{run_id}')
        current_serial = run.config.get('serial', None)

        # Refuse to move a run that is not where it was claimed to be. Without this, a typo in
        # --from_serial silently relabels an unrelated run, and W&B has no undo.
        if current_serial != from_serial:
            raise SystemExit(
                f'{run_id}: config.serial is {current_serial!r}, not {from_serial!r}. '
                f'Refusing to touch it -- check the run id and --from_serial.')

        new_name = renamed_run_name(run.name, from_serial, to_serial)
        tag = 'APPLY' if execute else 'DRY-RUN'
        name_line = new_name if new_name else f'(unchanged: no _{from_serial} suffix)'

        print(f'[{tag}] {run_id}  ({run.state}, created {run.created_at})')
        print(f'    serial : {current_serial}  ->  {to_serial}')
        print(f'    name   : {run.name}')
        print(f'         -> {name_line}')
        print(f'    model  : {run.config.get("model_id")}')
        print(f'    dataset: {run.config.get("dataset")}  split={run.config.get("split")}')

        if execute:
            run.config['serial'] = to_serial
            if new_name:
                run.name = new_name
            run.update()
            serial_updated += 1
            name_updated += 1 if new_name else 0

    print()
    if execute:
        print(f'Applied: {serial_updated} serial(s) and {name_updated} name(s) updated.')
    else:
        print(f'DRY RUN over {len(run_ids)} run(s). Nothing was written. '
              f'Pass --execute to apply.')
    return 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--project_name', type=str, default='LisTAya/LALMDecoder',
                   help='project_entity/project_name')
    p.add_argument('--run_ids', nargs='+', type=str, required=True,
                   help='Explicit W&B run ids. Required: duplicated cells are '
                        'indistinguishable by config, so there is no safe selector.')
    p.add_argument('--from_serial', type=int, required=True,
                   help='The serial the runs are currently under. Asserted before writing.')
    p.add_argument('--to_serial', type=int, required=True)
    p.add_argument('--execute', action='store_true',
                   help='Actually write to W&B. Without it this is a dry run.')
    return p.parse_args()


def main():
    args = parse_args()
    return rename_serial(args.project_name, args.run_ids, args.from_serial,
                         args.to_serial, args.execute)


if __name__ == '__main__':
    main()
