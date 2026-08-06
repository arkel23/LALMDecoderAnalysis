"""Regenerates the source-of-truth tables for what was trained and what is evaluated.

Everything here is DERIVED, from the wandb run configs or from a frozen snapshot -- nothing is
typed by hand. Three outputs:

  data/manifest_training.csv      one row per training cell: what it trained on and what its
                                  checkpoint was SELECTED on. The ha_ng collision -- its
                                  selection split is a config the eval sweeps also use -- is
                                  visible directly in this table.
  data/manifest_eval_sets.csv     the eval-dataset registry joined to the audio statistics the
                                  eval runs already log (num_samples, audio_length_s_*).
  data/language_hours_whisper.csv frozen snapshot of Whisper's pretraining hours per language,
                                  copied from MultilingualQASR. Never referenced live.

TinyAya's language roster and regions are NOT rebuilt here: they already exist in
data/tinyaya_report/tinyaya_language_composition.csv (69 languages, with report_region), written
by fetch_tinyaya_composition.py.

Run from the repo root: python build_manifests.py
"""
import os
import shutil

import pandas as pd

from utils import LANGUAGE_DIC, RESOURCE_TIER, LANGUAGE_REGION, is_selection_split, GRID_SERIAL

HIST = os.path.join('data', 'raw_serials', f'history_serial_{GRID_SERIAL}.csv')
REGISTRY = os.path.join('for_quantizedasr', 'tools', 'preprocess', 'eval_datasets.csv')
EVAL_SERIALS = [os.path.join('data', 'raw_serials', f'raw_serial_{s}.csv') for s in (10, 11)]
WHISPER_HOURS_SRC = '/home/edwinrios/analysis/MultilingualQASR/data/language_hours.csv'
OUT = 'data'
FLOAT_FORMAT = '%.6f'


def training_manifest():
    df = pd.read_csv(HIST)
    g = (df.groupby(['dataset', 'dataset_path_train', 'dataset_train', 'split_train',
                     'dataset_path', 'split'], dropna=False)
         .agg(n_runs=('run_id', 'nunique'),
              variants=('model_id', 'nunique'),
              seeds=('seed', lambda s: ','.join(str(x) for x in sorted(set(s)))))
         .reset_index()
         .rename(columns={'dataset': 'study_cell',
                          'dataset_path': 'eval_dataset_path',
                          'split': 'eval_split'}))
    g['eval_dataset'] = g['study_cell']
    g['language_name'] = g['study_cell'].map(LANGUAGE_DIC)
    g['resource_tier'] = g['study_cell'].map(RESOURCE_TIER)
    g['region'] = g['study_cell'].map(LANGUAGE_REGION)
    # A cell's training eval IS its selection split by definition, so recording that would be a
    # tautology. What matters is whether that split ALSO appears as an eval config the sweeps
    # would run -- for ha_ng it does, which is why it is excluded there.
    reg = pd.read_csv(REGISTRY)
    collide = {}
    for _, r in reg.iterrows():
        if is_selection_split(r['study_cell'], r['dataset_path'], r['dataset'], r['split']):
            collide[r['study_cell']] = r['config_yaml']
    g['selection_split_collides_with'] = g['study_cell'].map(collide).fillna('')
    cols = ['study_cell', 'language_name', 'region', 'resource_tier',
            'dataset_path_train', 'dataset_train', 'split_train',
            'eval_dataset_path', 'eval_dataset', 'eval_split',
            'selection_split_collides_with', 'variants', 'seeds', 'n_runs']
    return g[cols].sort_values('study_cell')


def eval_sets_manifest():
    reg = pd.read_csv(REGISTRY)
    frames = [pd.read_csv(f) for f in EVAL_SERIALS if os.path.exists(f)]
    if not frames:
        return reg
    runs = pd.concat(frames, ignore_index=True)
    stats = ['num_samples', 'audio_length_s_mean', 'audio_length_s_std',
             'audio_length_s_min', 'audio_length_s_max']
    have = [c for c in stats if c in runs.columns]
    # The eval runs already measure every eval set; take one finished row per set.
    obs = (runs[runs['state'] == 'finished']
           .dropna(subset=have, how='all')
           .drop_duplicates(['dataset_path', 'dataset', 'split'])
           [['dataset_path', 'dataset', 'split'] + have])
    return reg.merge(obs, on=['dataset_path', 'dataset', 'split'], how='left')


def main():
    os.makedirs(OUT, exist_ok=True)

    train = training_manifest()
    train.to_csv(os.path.join(OUT, 'manifest_training.csv'), index=False)
    n_col = int((train['selection_split_collides_with'] != '').sum())
    print(f'manifest_training.csv: {len(train)} cells, {n_col} whose selection split is also an '
          f'eval config the sweeps would otherwise run')
    print(train[['study_cell', 'dataset_train', 'eval_dataset_path', 'eval_split',
                 'selection_split_collides_with']].to_string(index=False))

    ev = eval_sets_manifest()
    ev.to_csv(os.path.join(OUT, 'manifest_eval_sets.csv'), index=False,
              float_format=FLOAT_FORMAT)
    n_stats = int(ev['audio_length_s_mean'].notna().sum()) if 'audio_length_s_mean' in ev else 0
    print(f'\nmanifest_eval_sets.csv: {len(ev)} eval sets, {int(ev["use_in_sweep"].sum())} swept, '
          f'{n_stats} with audio statistics')

    if os.path.exists(WHISPER_HOURS_SRC):
        dst = os.path.join(OUT, 'language_hours_whisper.csv')
        shutil.copyfile(WHISPER_HOURS_SRC, dst)
        h = pd.read_csv(dst)
        ours = {c.split('_')[0] for c in LANGUAGE_DIC}
        covered = ours & set(h['language_code'])
        print(f'\nlanguage_hours_whisper.csv: {len(h)} languages, covering '
              f'{len(covered)}/{len(ours)} of ours. Absent: {sorted(ours - covered)} '
              f'(Whisper does not support it -- that is the finding, not a gap).')
    else:
        print(f'\n[SKIP] {WHISPER_HOURS_SRC} not present')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
