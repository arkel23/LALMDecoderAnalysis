"""Which runs of an eval serial are missing, and a .sh to fill them.

Adapted from ChineseQASR's missing_scripts.py: same state lookup (finished wins over a later
failure) and same rerun rule (MISSING / FAILED / CRASHED). No quantisation axis here.

The expected grid is an explicit MODEL_CONFIGS x DATASET_CONFIGS cross product, both overridable
on the command line. Serial 10 is exactly that shape.

Serial 11 is NOT: each trained checkpoint is paired with only its own language's datasets, so a
cross product would invent runs that should never happen. Pass --pairings to read those pairings
from the sweep instead.

Model configs need their YAML for `model_id`, so --model_dir must point at a QuantizedASR
checkout. Dataset configs resolve offline -- this repo owns the generator that writes them.

Usage:
    python missing_runs.py --serial 10
    python missing_runs.py --serial 10 --model_configs whisper_medium.yaml
    python missing_runs.py --serial 11 \\
        --pairings for_quantizedasr/scripts/eval_lalm_decoder_txf.sh
"""
import os
import re
import glob
import argparse

import pandas as pd

from verify_eval_pairing import parse_pairings

RERUN_STATES = (None, 'failed', 'crashed')
QA_MODELS = '/home/edwinrios/projects/QuantizedASR/configs/models'

# The serial 10 baselines: the study's own encoder plus the two commercial LALMs. Pinned
# against eval_lalm_baselines.sh by test_utils_port.py.
MODEL_CONFIGS = [
    'whisper_medium.yaml',
    'voxtral_mini_3b.yaml',
    'qwen_2_audio_7b.yaml',
]

REGISTRY = os.path.join('for_quantizedasr', 'tools', 'preprocess', 'eval_datasets.csv')


def load_registry(path=REGISTRY, swept_only=True):
    """The eval-dataset registry: one row per config, with use_in_sweep derived.

    Single source of truth, shared with both sweeps. Reading it here is the point -- this list
    used to be typed out separately and drifted from the sweep by exactly the config that must
    not be evaluated.
    """
    d = pd.read_csv(path)
    return d[d['use_in_sweep']] if swept_only else d


def dataset_key(config, registry):
    """config_yaml -> (dataset_path, dataset, split), straight from the registry."""
    row = registry[registry['config_yaml'] == config]
    if row.empty:
        return None
    r = row.iloc[0]
    return (r['dataset_path'], r['dataset'], r['split'])


def model_id_of(config, model_dir):
    path = os.path.join(model_dir, os.path.basename(config))
    if not os.path.exists(path):
        return None
    for line in open(path):
        m = re.match(r"\s*model_id:\s*'?([^'\n]+)'?", line)
        if m:
            return m.group(1).strip()
    return None


def paired_grid(sweep, model_dir, registry):
    """Each trained checkpoint against only its own language's datasets, registry-filtered."""
    swept = set(registry['config_yaml'])
    text = open(sweep).read()
    variants = re.findall(r'[\w]+', re.search(r'^VARIANTS=\((.*?)\)', text, re.M).group(1))
    out = []
    for stem, cfgs in parse_pairings(sweep):
        for v in variants:
            hits = [h for h in sorted(glob.glob(os.path.join(
                model_dir, f'cq2a_whisper_medium_tiny_aya_{v}_txf_ws_{stem}_*.yaml')))
                if not h.endswith('_1k.yaml')]
            for h in hits:
                out += [(os.path.basename(h), c) for c in cfgs if c in swept]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--serial', type=int, default=10)
    p.add_argument('--input_file', type=str, default=None)
    p.add_argument('--model_configs', nargs='+', default=MODEL_CONFIGS)
    p.add_argument('--registry', type=str, default=REGISTRY)
    p.add_argument('--pairings', type=str, default=None,
                   help='Sweep script to read per-language pairings from, instead of a cross '
                        'product. Needed for serial 11.')
    p.add_argument('--model_dir', type=str, default=QA_MODELS)
    p.add_argument('--output_sh', type=str, default=None)
    p.add_argument('--batch_size', type=int, default=128)
    args = p.parse_args()

    raw = args.input_file or os.path.join('data', 'raw_serials', f'raw_serial_{args.serial}.csv')
    out_sh = args.output_sh or os.path.join('results_all', f'{args.serial}_missing.sh')
    if not os.path.exists(raw):
        raise SystemExit(f'{raw} not present -- download the serial first.')

    # Finished wins: a cell that failed once and later succeeded is done.
    state = {}
    for _, r in pd.read_csv(raw).iterrows():
        key = (r['dataset_path'], r['dataset'], r['split'], r['model_id'])
        if key not in state or r['state'] == 'finished':
            state[key] = r['state']

    registry = load_registry(args.registry)
    grid = (paired_grid(args.pairings, args.model_dir, registry) if args.pairings else
            [(m, d) for m in args.model_configs for d in registry['config_yaml']])

    lines, reasons, unresolved = [], [], []
    for model_cfg, ds_cfg in grid:
        mid, dkey = model_id_of(model_cfg, args.model_dir), dataset_key(ds_cfg, registry)
        if mid is None or dkey is None:
            unresolved.append((model_cfg, ds_cfg))
            continue
        st = state.get(dkey + (mid,))
        if st in RERUN_STATES:
            reasons.append(('MISSING' if st is None else st.upper(), model_cfg))
            lines.append(
                f'python -m tools.evaluate --serial {args.serial} '
                f'--batch_size {args.batch_size} --eval_metrics wer_all cer '
                f'--wandb_entity LisTAya --wandb_project LALMDecoder '
                f'--config configs/models/{model_cfg} configs/datasets/{ds_cfg}')

    if unresolved:
        print(f'[WARN] {len(unresolved)} pair(s) unresolved -- is --model_dir right? '
              f'e.g. {unresolved[0]}')

    by_model = {}
    for reason, m in reasons:
        by_model.setdefault(m, {}).setdefault(reason, 0)
        by_model[m][reason] += 1
    print(f'serial {args.serial}: {len(grid)} expected, {len(grid) - len(lines)} done, '
          f'{len(lines)} to run\n')
    for m in sorted(by_model):
        print(f'  {m:52s} {by_model[m]}')

    os.makedirs(os.path.dirname(out_sh) or '.', exist_ok=True)
    with open(out_sh, 'w') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))
    print(f'\nWrote {len(lines)} command(s) to {out_sh}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
