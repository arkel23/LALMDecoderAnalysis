"""
Guard: every trained checkpoint must only be evaluated on its OWN language's datasets.

Why this exists. Each TinyAya variant now has one checkpoint per training language --
`cq2a_whisper_medium_tiny_aya_fire_txf_ws_ur_pk_250.yaml` is the fire decoder trained on Urdu,
and it has no claim on Hausa. The original sweep shape in QuantizedASR's scripts/ is a full
model x dataset cross product, which is correct when every model config is language-agnostic and
silently wrong here: it would pair the Urdu-trained fire checkpoint with every other language and
report the results as though they meant something.

`eval_lalm_decoder_txf.sh` pairs correctly today -- this was verified by dry run (334
evaluations, zero cross-language pairings). The risk is drift: the PAIRINGS block is
hand-maintained, and every new language is an opportunity to mistype one. So the check that was
run once by hand runs on every `bash plotter.sh` instead.

It is a STATIC check. It parses the PAIRINGS block out of the sweep script and the manifest
emitted by create_yamls_worldspeech_lalm.py, and needs neither the generated YAMLs nor the
QuantizedASR checkout -- so it works from a bare clone of this repo.

The study cell is the unit of comparison, not the config name: `fr_ca` (trained) and `fr_fr`
(FLEURS eval) are the same cell, as are `es_mx` and `es_419`. Comparing raw config names would
flag those correct pairings as errors.

Usage:
    python verify_eval_pairing.py
Exit code is non-zero if any pairing crosses a language boundary.
"""
import os
import re
import sys
import csv
import argparse

from utils import to_study_cell, LANGUAGE_REGION

SWEEP = os.path.join('for_quantizedasr', 'scripts', 'eval_lalm_decoder_txf.sh')
# The PAIRINGS array no longer lives in the sweep -- it is generated next to the manifest by
# create_yamls_worldspeech_lalm.py and sourced. Checking the generated file is checking what
# the sweep actually runs.
PAIRINGS = os.path.join('for_quantizedasr', 'tools', 'preprocess',
                        'worldspeech_lalm_pairings.sh')
MANIFEST = os.path.join('for_quantizedasr', 'tools', 'preprocess',
                        'worldspeech_lalm_manifest.csv')

ok = True


def check(name, cond, detail=''):
    global ok
    ok = ok and bool(cond)
    print(f'[{"PASS" if cond else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))


def load_cell_map(manifest_path):
    """dataset config name -> study cell.

    The manifest is authoritative for the WorldSpeech side, because it is emitted by the same
    generator that writes the configs -- so held-out variants like `en_au` and `fr_cd`, which
    are not training configs and therefore absent from utils.TRAIN_CONFIG_TO_CELL, are covered.
    """
    cell = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            for row in csv.DictReader(fh):
                cell[row['dataset']] = row['study_cell']
    # FLEURS configs are named after the evaluated variety; to_study_cell handles the rest.
    for lang in LANGUAGE_REGION:
        cell.setdefault(lang, to_study_cell(lang))
    return cell


def parse_pairings(sweep_path):
    """Extract the PAIRINGS array as (train_stem, fleurs, [in_training], [held_out])."""
    text = open(sweep_path).read()
    block = re.search(r'PAIRINGS=\((.*?)\n\)', text, re.S)
    if not block:
        return None
    out = []
    for line in block.group(1).splitlines():
        line = line.strip()
        if not line.startswith('"'):
            continue
        parts = line.strip('"').split('|')
        if len(parts) < 2:
            continue
        stem = parts[0]
        fleurs = [parts[1]] if parts[1] else []
        ws_in = parts[2].split() if len(parts) > 2 and parts[2] else []
        ws_held = parts[3].split() if len(parts) > 3 and parts[3] else []
        out.append((stem, fleurs, ws_in, ws_held))
    return out


def config_to_dataset(path):
    """'short_ml/worldspeech_en_au_test.yaml' -> 'en_au'."""
    base = os.path.basename(path)
    m = re.match(r'(?:fleurs|worldspeech)_([a-z]{2,3}_[a-z0-9]{2,4})_(?:test|dev)\.yaml$', base)
    return m.group(1) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep', type=str, default=SWEEP)
    p.add_argument('--pairings', type=str, default=PAIRINGS)
    p.add_argument('--manifest', type=str, default=MANIFEST)
    args = p.parse_args()

    if not os.path.exists(args.sweep):
        print(f'[SKIP] {args.sweep} not present')
        return 0

    # One source of truth. A stale inline copy left behind in the sweep would shadow the
    # generated one on `source`-order, and the two could drift apart unnoticed -- so the sweep
    # must carry the source line and no array of its own.
    sweep_text = open(args.sweep).read()
    check('the sweep sources the generated pairings rather than inlining them',
          'worldspeech_lalm_pairings.sh' in sweep_text
          and not re.search(r'^PAIRINGS=\(', sweep_text, re.M))
    check(f'the generated pairings file exists ({args.pairings})',
          os.path.exists(args.pairings),
          'run create_yamls_worldspeech_lalm.py' if not os.path.exists(args.pairings) else '')
    if not os.path.exists(args.pairings):
        return 1

    cell_map = load_cell_map(args.manifest)
    pairings = parse_pairings(args.pairings)

    check('the PAIRINGS block parses', pairings is not None)
    if not pairings:
        return 1
    check('every study language has a pairing', len(pairings) >= 10, f'{len(pairings)} pairings')

    n_pairs, crossings, unmapped = 0, [], []
    seen_cells = set()

    for stem, fleurs, ws_in, ws_held in pairings:
        model_cell = cell_map.get(stem, to_study_cell(stem))
        seen_cells.add(model_cell)
        for cfg in fleurs + ws_in + ws_held:
            ds = config_to_dataset(cfg)
            if ds is None:
                unmapped.append(cfg)
                continue
            data_cell = cell_map.get(ds, to_study_cell(ds))
            n_pairs += 1
            if data_cell != model_cell:
                crossings.append((stem, model_cell, ds, data_cell))

    check('every dataset config name is parseable', not unmapped,
          '; '.join(unmapped[:4]))
    check(f'no pairing crosses a language boundary ({n_pairs} checked)',
          not crossings,
          '; '.join(f'model {s} ({mc}) x data {d} ({dc})'
                    for s, mc, d, dc in crossings[:4]))

    # A pairing that trains on two configs must evaluate on both, or the second is silently
    # unused -- which is how ta_lk went unevaluated in the first version of the generator.
    if os.path.exists(args.manifest):
        with open(args.manifest) as fh:
            rows = list(csv.DictReader(fh))
        listed = {c for _, f, i, h in pairings for c in f + i + h}
        listed_ds = {config_to_dataset(c) for c in listed}
        missing = [r['dataset'] for r in rows if r['dataset'] not in listed_ds]
        check('every generated WorldSpeech config appears in some pairing',
              not missing, f'unused: {", ".join(missing[:8])}')

        held = [r['dataset'] for r in rows if r['status'] == 'held_out']
        check('the held-out variants are all reachable via --eval_set all',
              all(d in listed_ds for d in held),
              f'{sum(1 for d in held if d not in listed_ds)} unreachable')

    print(f'\n{"PAIRING OK" if ok else "PAIRING BROKEN"} '
          f'({len(pairings)} language cells, {n_pairs} model-dataset pairings)')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
