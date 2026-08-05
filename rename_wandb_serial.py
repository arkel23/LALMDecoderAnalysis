"""Moves specific W&B runs to a different `serial` in place, keeping the run NAME in sync.

Selection is by explicit --run_ids, not by a config condition: the case this exists for is a
re-run, where old and new share every config field and only the run id differs.

Run names bake in a trailing "_<serial>" that does not follow config.serial, so it is rewritten
too.

MUTATES a live W&B project, so it is a DRY RUN by default. Pass --execute to apply.

    python rename_wandb_serial.py --run_ids n4cot5v7 --from_serial 0 --to_serial 1
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
