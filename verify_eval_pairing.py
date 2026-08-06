"""
Guard: every trained checkpoint is evaluated only on its own language.

Each checkpoint is language-specific, so the usual model x dataset cross product is wrong here
-- it would pair the Urdu-trained connector with Hausa and report the result as meaningful.
Static check over the sweep's PAIRINGS block; needs no QuantizedASR checkout.

Usage: python verify_eval_pairing.py
"""
import os
import re
import sys
import argparse

SWEEP = os.path.join('for_quantizedasr', 'scripts', 'eval_lalm_decoder_txf.sh')
BASELINES = os.path.join('for_quantizedasr', 'scripts', 'eval_lalm_baselines.sh')
GENERATOR = os.path.join('for_quantizedasr', 'tools', 'preprocess',
                         'create_yamls_worldspeech_lalm.py')
REGISTRY = os.path.join('for_quantizedasr', 'tools', 'preprocess', 'eval_datasets.csv')


FAILURES = []


def check(name, cond, detail=''):
    """Records the failure as well as printing it -- a guard that prints FAIL and exits 0 is
    worse than no guard, and that had happened here."""
    if not cond:
        FAILURES.append(name)
    print(f'[{"PASS" if cond else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))
    return bool(cond)


def language(config):
    """fr_ca trains, fleurs_fr_fr evaluates; es_mx trains, fleurs_es_419 evaluates. Comparing
    the language prefix accepts those and still rejects ur_pk x ha_td."""
    return config.split('_')[0]


def generated_worldspeech_configs():
    """The generator's `configs` list -- the authority on which variants exist."""
    block = re.search(r'^configs = \[(.*?)^\]',
                      open(GENERATOR).read(), re.S | re.M)
    return re.findall(r'"([^"]+)"', block.group(1)) if block else []


def parse_pairings(path):
    """-> [(train_stem, [dataset configs])]"""
    block = re.search(r'PAIRINGS=\((.*?)\n\)', open(path).read(), re.S)
    if not block:
        return []
    out = []
    for line in block.group(1).splitlines():
        line = line.strip()
        if line.startswith('"'):
            stem, *fields = line.strip('"').split('|')
            out.append((stem, ' '.join(fields).split()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep', type=str, default=SWEEP)
    p.add_argument('--baselines', type=str, default=BASELINES)
    args = p.parse_args()

    if not os.path.exists(args.sweep):
        print(f'[SKIP] {args.sweep} not present')
        return 0

    pairings = parse_pairings(args.sweep)
    if not pairings:
        print('[FAIL] no PAIRINGS block found')
        return 1

    n, crossings = 0, []
    for stem, configs in pairings:
        for config in configs:
            n += 1
            dataset = os.path.basename(config).replace('.yaml', '')
            dataset = re.sub(r'^(fleurs|worldspeech)_|_(test|dev)$', '', dataset)
            if language(dataset) != language(stem):
                crossings.append(f'{stem} x {dataset}')

    if crossings:
        print(f'[FAIL] {len(crossings)} pairing(s) cross a language boundary: '
              f'{", ".join(crossings[:4])}')
        return 1

    # Coverage is now a property of the REGISTRY, not of either script: both read from it, so
    # grepping them for config names would check nothing. Every variant of a study language must
    # be in the registry, and swept unless it is a selection split.
    study = {language(stem) for stem, _ in pairings}
    want = {c for c in generated_worldspeech_configs() if language(c) in study}
    if os.path.exists(REGISTRY):
        import csv as _csv
        with open(REGISTRY) as fh:
            reg = list(_csv.DictReader(fh))
        listed = {r['dataset'] for r in reg if r['source'] == 'worldspeech'}
        check(f'the registry covers every study-language WorldSpeech variant ({len(want)})',
              want <= listed, ', '.join(sorted(want - listed)[:6]))

    # The sweep must consume every field of every pairing. It once defaulted to a subset flag
    # and silently ran 26 of the 44 datasets per variant -- the guard already knew the number
    # was 44, it just was not checking that the sweep used them all.
    body = open(args.sweep).read().split('PAIRINGS=', 1)[-1]
    uses_all = 'for c in $fleurs_cfg $ws_in $ws_held' in body
    no_subset = 'eval_set' not in body
    check('the sweep evaluates every pairing field, with no subset flag',
          uses_all and no_subset,
          '' if uses_all and no_subset else
          f'uses_all={uses_all} no_subset_flag={no_subset}')
    if not (uses_all and no_subset):
        return 1

    # The registry decides membership. A config that is some cell's training-time selection
    # split is not a held-out eval, and must not reach either sweep.
    if os.path.exists(REGISTRY):
        import csv as _csv
        from utils import is_selection_split
        with open(REGISTRY) as fh:
            reg = list(_csv.DictReader(fh))
        swept = {r['config_yaml'] for r in reg if r['use_in_sweep'] == 'True'}
        bad = [r['config_yaml'] for r in reg
               if r['use_in_sweep'] == 'True'
               and is_selection_split(r['study_cell'], r['dataset_path'],
                                      r['dataset'], r['split'])]
        check('no swept config is a training-time selection split', not bad,
              ', '.join(bad[:3]))
        excluded = [r['config_yaml'] for r in reg if r['use_in_sweep'] != 'True']
        check(f'the registry excludes exactly the selection splits ({len(excluded)})',
              all(is_selection_split(r['study_cell'], r['dataset_path'], r['dataset'],
                                     r['split'])
                  for r in reg if r['config_yaml'] in excluded))

        # Every config either sweep would run must come from the registry.
        for path in (args.sweep, args.baselines):
            if not os.path.exists(path):
                continue
            refs = set(re.findall(r'(short_ml/[a-z0-9_]+\.yaml)', open(path).read()))
            check(f'{os.path.basename(path)} names no config outside the swept registry',
                  refs <= swept, ', '.join(sorted(refs - swept)[:4]))
        if bad:
            return 1

    if FAILURES:
        print(f'\nPAIRING BROKEN -- {len(FAILURES)} check(s) failed')
        return 1

    print(f'PAIRING OK ({len(pairings)} cells, {n} datasets per variant, '
          f'{len(want)} WorldSpeech variants in both sweeps)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
