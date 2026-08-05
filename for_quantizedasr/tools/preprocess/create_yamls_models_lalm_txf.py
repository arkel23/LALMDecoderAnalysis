"""
Generates configs/models/*_txf_*.yaml for the trained connector checkpoints, by crawling the
ERISLab HF organisation. `txf` marks a checkpoint trained by us, as opposed to an off-the-shelf
composition with a randomly-initialised connector.

Two checkpoints exist per cell -- the best step by eval CER and step 1000 -- and both get a
config, since the study reports best as primary and final as secondary.

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
CKPT_RE = re.compile(
    r'^ERISLab/q2a_(?P<encoder>.+?)_(?P<decoder>CohereLabs_tiny-aya-\w+)'
    r'_(?P<dataset>[a-z]+)-(?P<lang>[a-z]{2,3}_[a-z]{2,3})-(?P<step>\d+)$')

# es_es was re-run from es_mx; its checkpoints remain on the Hub but evaluating them would
# measure a condition no longer in the study.
SUPERSEDED_LANGS = ('es_es',)
INCLUDE_SUPERSEDED = False

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
n_written, n_skipped = 0, 0

for ckpt in sorted(checkpoints, key=lambda c: (c['lang'], c['decoder'], int(c['step']))):
    if ckpt['lang'] in SUPERSEDED_LANGS and not INCLUDE_SUPERSEDED:
        n_skipped += 1
        continue

    variant = ckpt['decoder'].rsplit('-', 1)[-1]
    step = int(ckpt['step'])
    step_label = f'{step // 1000}k' if step >= 1000 and step % 1000 == 0 else str(step)
    filename = (f"cq2a_whisper_medium_tiny_aya_{variant}_txf_"
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
variants = sorted({c['decoder'].rsplit('-', 1)[-1] for c in checkpoints})
print(f'\nGenerated {n_written} model config files ({n_skipped} superseded skipped).')
print(f'Coverage: {len(langs)} languages x up to {len(variants)} variants')

for lang in langs:
    have = {c['decoder'].rsplit('-', 1)[-1] for c in checkpoints if c['lang'] == lang}
    notes = ([f'no {"/".join(sorted(set(variants) - have))}'] if have != set(variants) else [])
    if not any(int(c['step']) == 1000 for c in checkpoints if c['lang'] == lang):
        notes.append('no step-1000 checkpoint')
    if notes:
        print(f'  {lang}: {"; ".join(notes)}')

for expected in ('am_et', 'crs_sc'):
    if expected not in langs:
        print(f'  {expected}: no checkpoints uploaded (W&B runs exist)')
