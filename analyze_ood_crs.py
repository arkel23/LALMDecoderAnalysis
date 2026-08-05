"""The crs_sc (Seychellois Creole) cell: the out-of-distribution probe.

Supported by neither the Whisper encoder nor any TinyAya decoder, so it is the only cell where
both halves operate outside their coverage. Also the only language with all six models,
including the non-Aya Qwen3-4B control -- the one place the question "TinyAya property or
connector-recipe property?" can be asked.

Coverage is the limitation, not data: the WorldSpeech report gives crs_sc 1,602 hours. Its
train/val/test splits all carry the duration-consistency cleaning.

Usage:
    python analyze_ood_crs.py --input_file results_all/acc/t1_sample_efficiency.csv \
        --output_file results_all/acc/t3_crs_ood.csv
"""
import os
import argparse
import pandas as pd

from utils import MODEL_FAMILY, assert_unique_keys


OOD_LANGUAGE = 'crs_sc'
FLOAT_FORMAT = '%.6f'

EVAL_SPLIT_NOTE = (
    "val_clean carries the same duration-consistency cleaning as test_clean: samples whose "
    "decoded audio length disagrees with the corpus duration column by >=1s are removed"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input_file', type=str,
                   default=os.path.join('results_all', 'acc', 't1_sample_efficiency.csv'))
    p.add_argument('--output_file', type=str,
                   default=os.path.join('results_all', 'acc', 't3_crs_ood.csv'))
    p.add_argument('--language', type=str, default=OOD_LANGUAGE)
    args = p.parse_args()

    df = pd.read_csv(args.input_file)
    # See analyze_region_match.py: crs_sc is precisely the cell being re-run, so without this
    # the OOD table would carry both the completed run and its half-trained replacement.
    if 'is_canonical' in df.columns:
        df = df[df['is_canonical']]
    out = df[df['dataset'] == args.language].copy()
    if out.empty:
        raise SystemExit(f'No rows for {args.language} in {args.input_file}')

    assert_unique_keys(out, ['model_id'], label=f't3 {args.language}')

    out['model_family'] = out['model_id'].map(MODEL_FAMILY)
    out['eval_split_note'] = EVAL_SPLIT_NOTE

    # Spread across decoders, on the metric that is not an arbitrary stopping point.
    finished = out[out['state'] == 'finished']
    out['best_cer_spread_finished'] = (
        finished['best_cer'].max() - finished['best_cer'].min() if len(finished) else float('nan')
    )

    keep = ['dataset', 'model_id', 'model_short', 'model_family', 'state', 'n_evals',
            'best_cer', 'final_cer', 'final_minus_best', 'audio_h_to_best',
            'audio_h_to_1.5x_best', 'late_sd', 'audio_h_total',
            'best_cer_spread_finished', 'language_status', 'split', 'eval_split_note']
    out = out[[c for c in keep if c in out.columns]].sort_values('best_cer')

    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    out.to_csv(args.output_file, index=False, float_format=FLOAT_FORMAT)
    print(f'Wrote {len(out)} rows to {args.output_file}\n')
    print(out[['model_short', 'model_family', 'state', 'best_cer', 'final_cer',
               'final_minus_best', 'late_sd']].to_string(index=False))
    return 0


if __name__ == '__main__':
    main()
