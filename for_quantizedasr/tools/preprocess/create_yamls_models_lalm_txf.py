"""
Generates `configs/models/*_txf_*.yaml` for the LALM decoder study's trained connector
checkpoints, by crawling the ERISLab HuggingFace organisation.

`txf` in a model-config name marks a checkpoint TRAINED BY US and uploaded to HF, as opposed
to an off-the-shelf composition of a stock encoder and a stock decoder. The untrained configs
(`cq2a_whisper_medium_tiny_aya_{base,global,earth,fire,water}.yaml`) build the model from
`openai/whisper-medium` plus a CohereLabs decoder with a randomly-initialised connector; the
configs generated here instead load a connector that was actually trained, so they are what an
evaluation sweep should point at.

WHERE THIS LIVES. Written in QuantizedASR's `tools/preprocess/` style but kept in
LALMDecoderAnalysis under `for_quantizedasr/`, because QuantizedASR is not modified from here.
Copy `for_quantizedasr/tools/` and `for_quantizedasr/scripts/` into that checkout and run this
from its root; `output_dir` is relative and resolves there, exactly like the generators it
mirrors.

CHECKPOINT NAMING on the Hub:

    ERISLab/q2a_{encoder}_{decoder}_{dataset}-{lang}-{step}
    e.g. ERISLab/q2a_openai_whisper-medium_CohereLabs_tiny-aya-earth_ws-ur_pk-250

`ws` abbreviates WorldSpeech, `{lang}` is the TRAINING config (so `es_mx` and `fr_ca`, not the
`es_419`/`fr_fr` those cells are evaluated on), and `{step}` is the training step. Two
checkpoints exist per cell: the best step by eval CER, and the end of training at step 1000.
Both get a config, because the study reports best-checkpoint CER as primary and final as
secondary -- evaluating only one of them would silently answer a different question.

Generated config name:  cq2a_whisper_medium_tiny_aya_{variant}_txf_ws_{lang}_{step}.yaml
with step 1000 written as `1k`, matching the existing txf configs' `_1k` / `_5k` / `_8k` style.

TWO GAPS THIS CRAWL EXPOSES, reported rather than silently skipped:
  * `es_es` has only best-step checkpoints and no step-1000. Those are the Spain-Spanish runs
    that were deleted from W&B on 2026-08-01 and re-run from `es_mx`. They are SKIPPED by
    default -- generating configs for them would point an eval at a superseded condition. Set
    INCLUDE_SUPERSEDED = True to emit them anyway.
  * `am_et` (Amharic) and `crs_sc` (Seychellois Creole) have NO checkpoints uploaded at all,
    although both have finished W&B runs. Nothing can be generated for them until they are
    pushed.

Usage (from the QuantizedASR repo root, after copying):
    python tools/preprocess/create_yamls_models_lalm_txf.py
"""
import os
import re
import json
import urllib.parse
import urllib.request

import yaml

HF_AUTHOR = 'ERISLab'
HF_API = 'https://huggingface.co/api/models'

# ERISLab/q2a_{encoder}_{decoder}_{dataset}-{lang}-{step}
CKPT_RE = re.compile(
    r'^ERISLab/q2a_(?P<encoder>.+?)_(?P<decoder>CohereLabs_tiny-aya-\w+)'
    r'_(?P<dataset>[a-z]+)-(?P<lang>[a-z]{2,3}_[a-z]{2,3})-(?P<step>\d+)$')

# Training configs whose W&B runs were deleted and superseded. Their checkpoints remain on the
# Hub, but pointing an eval at them would measure a condition that is no longer part of the
# study.
SUPERSEDED_LANGS = ('es_es',)
INCLUDE_SUPERSEDED = False

# The training runs used bfloat16 (`model_dtype: 'bfloat16'` in their W&B config). The txf
# configs in this repo all evaluate in float32, which is the higher-precision choice and costs
# nothing at eval time; kept consistent with them rather than with the training dtype.
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

    found, skipped = [], []
    for m in models:
        match = CKPT_RE.match(m['id'])
        if match:
            found.append({'repo_id': m['id'], **match.groupdict()})
        else:
            skipped.append(m['id'])
    return found, skipped


def short_variant(decoder):
    """'CohereLabs_tiny-aya-earth' -> 'earth'."""
    return decoder.rsplit('-', 1)[-1]


def step_label(step):
    """1000 -> '1k', matching the existing txf configs. Sub-1000 steps stay literal."""
    step = int(step)
    return f'{step // 1000}k' if step >= 1000 and step % 1000 == 0 else str(step)


def original_model_id(encoder, decoder):
    """The untrained composition this checkpoint was fine-tuned from.

    Emitted as a leading comment, which is the convention every existing txf config follows --
    it records what the checkpoint IS, since the ERISLab repo name alone does not make the
    encoder/decoder pairing obvious at a glance.
    """
    return f"q2a_{encoder.replace('_', '/', 1)}_{decoder.replace('_', '/', 1)}"


def main():
    checkpoints, skipped = fetch_checkpoints()
    print(f'Crawled {HF_AUTHOR}: {len(checkpoints)} checkpoints matched, '
          f'{len(skipped)} repos did not match the naming pattern')
    for repo in skipped:
        print(f'  unmatched: {repo}')

    output_dir = 'configs/models'
    os.makedirs(output_dir, exist_ok=True)

    n_written, n_skipped = 0, 0
    for ckpt in sorted(checkpoints, key=lambda c: (c['lang'], c['decoder'], int(c['step']))):
        if ckpt['lang'] in SUPERSEDED_LANGS and not INCLUDE_SUPERSEDED:
            n_skipped += 1
            continue

        variant = short_variant(ckpt['decoder'])
        filename = (f"cq2a_whisper_medium_tiny_aya_{variant}_txf_"
                    f"{ckpt['dataset']}_{ckpt['lang']}_{step_label(ckpt['step'])}.yaml")
        filepath = os.path.join(output_dir, filename)

        yaml_data = {
            'model_id': QuotedStr(ckpt['repo_id']),
            'local_model': QuotedStr(ckpt['repo_id']),
            'max_new_tokens': 200,
            'max_input_length': 30,
            'model_dtype': QuotedStr(MODEL_DTYPE),
        }

        with open(filepath, 'w') as fh:
            fh.write(f"# model_id: '{original_model_id(ckpt['encoder'], ckpt['decoder'])}'\n")
            yaml.dump(yaml_data, fh, default_flow_style=False, sort_keys=False)

        n_written += 1
        print(f'Created: {filename}')

    print(f'\nSuccessfully generated {n_written} model config files.')
    if n_skipped:
        print(f'Skipped {n_skipped} superseded checkpoint(s) for {SUPERSEDED_LANGS} '
              f'(set INCLUDE_SUPERSEDED = True to emit them).')

    # Report coverage gaps rather than leaving them to be discovered later.
    langs = sorted({c['lang'] for c in checkpoints})
    variants = sorted({short_variant(c['decoder']) for c in checkpoints})
    print(f'\nCoverage: {len(langs)} training languages x up to {len(variants)} variants')
    print(f'  languages: {", ".join(langs)}')
    for lang in langs:
        steps = {short_variant(c['decoder']): int(c['step'])
                 for c in checkpoints if c['lang'] == lang}
        missing = [v for v in variants if v not in steps]
        no_final = not any(int(c['step']) == 1000
                           for c in checkpoints if c['lang'] == lang)
        notes = []
        if missing:
            notes.append(f'no {"/".join(missing)}')
        if no_final:
            notes.append('NO step-1000 checkpoint')
        if notes:
            print(f'  {lang}: {"; ".join(notes)}')

    for expected in ('am_et', 'crs_sc'):
        if expected not in langs:
            print(f'  {expected}: NO checkpoints uploaded at all (W&B runs exist)')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
