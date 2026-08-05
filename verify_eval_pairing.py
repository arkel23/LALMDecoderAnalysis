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

    # Wrong pairings were guarded; missing ones were not, and the baselines sweep silently
    # covered 16 of 33 WorldSpeech variants. Every variant of a study language must appear in
    # both sweeps -- a checkpoint exists only for the trained variety, but evaluating the rest
    # is the accent-transfer axis and costs only inference.
    study = {language(stem) for stem, _ in pairings}
    want = {c for c in generated_worldspeech_configs() if language(c) in study}
    missing = []
    for path in (args.sweep, args.baselines):
        if not os.path.exists(path):
            continue
        have = set(re.findall(r'worldspeech_(.+?)_test\.yaml', open(path).read()))
        if want - have:
            missing.append(f'{os.path.basename(path)} misses '
                           f'{len(want - have)}: {", ".join(sorted(want - have)[:6])}')
    if missing:
        print('[FAIL] ' + '; '.join(missing))
        return 1

    print(f'PAIRING OK ({len(pairings)} cells, {n} model-dataset pairings, '
          f'{len(want)} WorldSpeech variants in both sweeps)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
