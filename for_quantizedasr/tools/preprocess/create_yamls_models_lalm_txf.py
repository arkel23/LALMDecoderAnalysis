"""
Generates configs/models/*_txf_*.yaml for the trained connector checkpoints, by crawling the
ERISLab HF organisation. `txf` marks a checkpoint trained by us, as opposed to an off-the-shelf
composition with a randomly-initialised connector.

Emits only what the eval sweep needs: the BEST checkpoint of each of the four grid-wide
variants. Step 1000 and the two control decoders are on the Hub but skipped -- flip
INCLUDE_FINAL / INCLUDE_CONTROLS to emit them.

Usage (from the QuantizedASR repo root):
    python tools/preprocess/create_yamls_models_lalm_txf.py
"""
import os
import re
import json
import urllib.parse
import urllib.request

import yaml

HF_API = 'https://huggingface.co/api/models'
HF_AUTHOR = 'ERISLab'

# ERISLab/q2a_openai_whisper-medium_CohereLabs_tiny-aya-earth_ws-ur_pk-250
# The decoder is org_name, and it is not always a TinyAya variant -- Qwen3-4B is the non-Aya
# control. Matching only tiny-aya silently dropped its four checkpoints.
CKPT_RE = re.compile(
    r'^ERISLab/q2a_(?P<encoder>[^_]+_[^_]+)_(?P<decoder>[^_]+_[\w.-]+)'
    r'_(?P<dataset>[a-z]+)-(?P<lang>[a-z]{2,3}_[a-z]{2,3})-(?P<step>\d+)$')


def slug(org_name):
    """'CohereLabs_tiny-aya-earth' -> 'tiny_aya_earth'; 'openai_whisper-medium' ->
    'whisper_medium'; 'Qwen_Qwen3-4B' -> 'qwen3_4b'."""
    return org_name.split('_', 1)[1].replace('-', '_').lower()

# es_es was re-run from es_mx; its checkpoints remain on the Hub but evaluating them would
# measure a condition no longer in the study.
SUPERSEDED_LANGS = ('es_es',)
INCLUDE_SUPERSEDED = False

# Two checkpoints exist per cell: the best step and step 1000. Only the best is evaluated --
# step 1000 is the overfitted end of the curve, and the best-to-final gap is already read off
# the training curves in t1. "Best" is whichever step is not 1000; verified against the Hub,
# every cell has exactly two and the best is always below 1000.
FINAL_STEP = 1000
INCLUDE_FINAL = False

# base and Qwen3-4B are controls that exist for only 2 of the 13 languages, so they are not part
# of the eval sweep. Kept configurable because they are what the crs_sc OOD contrast rests on.
CONTROL_DECODERS = ('tiny_aya_base', 'qwen3_4b')
INCLUDE_CONTROLS = False

# The existing txf configs all evaluate in float32, though training used bfloat16.
MODEL_DTYPE = 'float32'


class QuotedStr(str):
    pass


def quoted_scalar(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")


yaml.add_representer(QuotedStr, quoted_scalar)


def fetch_checkpoints():
    url = f'{HF_API}?' + urllib.parse.urlencode({'author': HF_AUTHOR, 'limit': 1000})
    with urllib.request.urlopen(url, timeout=120) as resp:
        models = json.load(resp)

    found = []
    for m in models:
        match = CKPT_RE.match(m['id'])
        if match:
            found.append({'repo_id': m['id'], **match.groupdict()})
    return found


output_dir = 'configs/models'
os.makedirs(output_dir, exist_ok=True)

checkpoints = fetch_checkpoints()
n_written = 0
skipped = {'superseded lang': 0, 'final step': 0, 'control decoder': 0}

for ckpt in sorted(checkpoints, key=lambda c: (c['lang'], c['decoder'], int(c['step']))):
    if ckpt['lang'] in SUPERSEDED_LANGS and not INCLUDE_SUPERSEDED:
        skipped['superseded lang'] += 1
        continue
    if int(ckpt['step']) == FINAL_STEP and not INCLUDE_FINAL:
        skipped['final step'] += 1
        continue
    if slug(ckpt['decoder']) in CONTROL_DECODERS and not INCLUDE_CONTROLS:
        skipped['control decoder'] += 1
        continue

    step = int(ckpt['step'])
    step_label = f'{step // 1000}k' if step >= 1000 and step % 1000 == 0 else str(step)
    filename = (f"cq2a_{slug(ckpt['encoder'])}_{slug(ckpt['decoder'])}_txf_"
                f"{ckpt['dataset']}_{ckpt['lang']}_{step_label}.yaml")

    yaml_data = {
        'model_id': QuotedStr(ckpt['repo_id']),
        'local_model': QuotedStr(ckpt['repo_id']),
        'max_new_tokens': 200,
        'max_input_length': 30,
        'model_dtype': QuotedStr(MODEL_DTYPE),
    }

    with open(os.path.join(output_dir, filename), 'w') as fh:
        # The untrained composition this was fine-tuned from, matching the existing txf configs.
        fh.write(f"# model_id: 'q2a_{ckpt['encoder'].replace('_', '/', 1)}"
                 f"_{ckpt['decoder'].replace('_', '/', 1)}'\n")
        yaml.dump(yaml_data, fh, default_flow_style=False, sort_keys=False)

    n_written += 1
    print(f'Created: {filename}')

langs = sorted({c['lang'] for c in checkpoints})
variants = sorted({slug(c['decoder']) for c in checkpoints})
print(f'\nGenerated {n_written} model config files from {len(checkpoints)} Hub checkpoints.')
for reason, n in skipped.items():
    if n:
        print(f'  skipped {n:3d}  {reason}')
print(f'Hub coverage: {len(langs)} languages x up to {len(variants)} decoders')

for lang in langs:
    have = {slug(c['decoder']) for c in checkpoints if c['lang'] == lang}
    notes = ([f'no {"/".join(sorted(set(variants) - have))}'] if have != set(variants) else [])
    if not any(int(c['step']) == 1000 for c in checkpoints if c['lang'] == lang):
        notes.append('no step-1000 checkpoint')
    if notes:
        print(f'  {lang}: {"; ".join(notes)}')

for expected in ('am_et', 'crs_sc'):
    if expected not in langs:
        print(f'  {expected}: no checkpoints uploaded (W&B runs exist)')
