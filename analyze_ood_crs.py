"""
The crs_sc (Seychellois Creole) cell: the out-of-distribution probe.

Why this language is treated separately rather than folded into the region contrast.
Seychellois Creole is an African language, but it is officially supported by NEITHER the
Whisper encoder NOR any TinyAya decoder variant. Every other language in the grid is seen
by at least one component. So crs_sc is the only cell where both halves of the model are
operating outside their training coverage -- the "neither" quadrant of the coverage matrix
the project's own plan asked for, already run.

It is also the only language with all six models: the four regional/global variants plus
tiny-aya-base and a non-Aya Qwen3-4B control. That control matters, because it is the only
place in the entire grid where the question "is this a TinyAya property or a
connector-recipe property?" can be asked at all.

Data availability is not the limitation here: the WorldSpeech report (arXiv 2605.09167)
gives crs_sc 1,602 hours, and the runs process ~2,100. What is missing is coverage, not data.

One caveat this script records rather than hides: the crs_sc runs evaluate on
ERISLab/WorldSpeech split 'val_clean'. The upstream cleaning script
(tools/preprocess/clean_worldspeech_duration_length_inconsistent.py) applies the
duration-consistency filter to the 'test' split only -- 'val_clean' and
'train_val_exc_clean' are an unfiltered 0.1%/99.9% re-split that merely materialises the
audio_length_s column. So the split's name promises a filter it did not receive.

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

EVAL_SPLIT_CAVEAT = (
    "val_clean is NOT duration-filtered: the upstream cleaning pass filters the 'test' "
    "split only; val_clean is an unfiltered 0.1% re-split of train with audio_length_s "
    "materialised"
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
    out = df[df['dataset'] == args.language].copy()
    if out.empty:
        raise SystemExit(f'No rows for {args.language} in {args.input_file}')

    assert_unique_keys(out, ['model_id'], label=f't3 {args.language}')

    out['model_family'] = out['model_id'].map(MODEL_FAMILY)
    out['eval_split_caveat'] = EVAL_SPLIT_CAVEAT

    # Spread across decoders, on the metric that is not an arbitrary stopping point.
    finished = out[out['state'] == 'finished']
    out['best_cer_spread_finished'] = (
        finished['best_cer'].max() - finished['best_cer'].min() if len(finished) else float('nan')
    )

    keep = ['dataset', 'model_id', 'model_short', 'model_family', 'state', 'n_evals',
            'best_cer', 'final_cer', 'final_minus_best', 'audio_h_to_best',
            'audio_h_to_1.5x_best', 'late_sd', 'audio_h_total',
            'best_cer_spread_finished', 'language_status', 'split', 'eval_split_caveat']
    out = out[[c for c in keep if c in out.columns]].sort_values('best_cer')

    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    out.to_csv(args.output_file, index=False, float_format=FLOAT_FORMAT)
    print(f'Wrote {len(out)} rows to {args.output_file}\n')
    print(out[['model_short', 'model_family', 'state', 'best_cer', 'final_cer',
               'final_minus_best', 'late_sd']].to_string(index=False))
    return 0


if __name__ == '__main__':
    main()
