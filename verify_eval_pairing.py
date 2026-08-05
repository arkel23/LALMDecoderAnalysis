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


def language(config):
    """fr_ca trains, fleurs_fr_fr evaluates; es_mx trains, fleurs_es_419 evaluates. Comparing
    the language prefix accepts those and still rejects ur_pk x ha_td."""
    return config.split('_')[0]


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

    print(f'PAIRING OK ({len(pairings)} cells, {n} model-dataset pairings)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
