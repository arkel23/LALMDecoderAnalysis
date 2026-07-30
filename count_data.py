import os
import argparse
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str, 
                        default=os.path.join('data', 'qasr_400.csv'),
                        help='filename for input .csv file from wandb')
    
    parser.add_argument('--main_serials', nargs='+', type=int, default=[23, 24])

    parser.add_argument('--output_file', default='count_model_and_status.csv', type=str,
                        help='File path')
    parser.add_argument('--results_dir', type=str, default='results_all',
                        help='The directory where results will be stored')
    
    parser.add_argument('--quant_type', action='store_true',
                        help='Use quant_type')

    args= parser.parse_args()
    return args

def main():
    args = parse_args()
    df = pd.read_csv(args.input_file)

    if not args.quant_type:
        print(df[['dataset_path', 'dataset', 'split']].drop_duplicates())
        num_groups = df[['dataset_path', 'dataset', 'split']].drop_duplicates().shape[0]
        print(num_groups)

        counts = (
            df.groupby(['serial', 'dataset_path', 'dataset', 'split'])['model_id']
            .agg(
                num_models='nunique',
                models=lambda x: list(pd.unique(x))
            )
        )

        print(counts)

        summary = (
        df.groupby(['serial', 'dataset_path', 'dataset', 'split', 'state'])
        .size()
        .unstack(fill_value=0)
        )

        print(summary)

    else:
        print(df[['dataset_path', 'dataset', 'split', 'quant_config', 'quant_dtype_weights']].drop_duplicates())
        num_groups = df[['dataset_path', 'dataset', 'split', 'quant_config', 'quant_dtype_weights']].drop_duplicates().shape[0]
        print(num_groups)

        counts = (
            df.groupby(['serial', 'dataset_path', 'dataset', 'split', 'quant_config', 'quant_dtype_weights'], dropna=False)['model_id']
            .agg(
                num_models='nunique',
                models=lambda x: list(pd.unique(x))
            )
        )

        print(counts)

        summary = (
        df.groupby(['serial', 'dataset_path', 'dataset', 'split', 'quant_config', 'quant_dtype_weights', 'state'], dropna=False)
        .size()
        .unstack(fill_value=0)
        )

        print(summary)
    

    final = summary.join(counts)
    final = final.reset_index()

    os.makedirs(args.results_dir, exist_ok=True)
    args.output_file = os.path.join(args.results_dir, args.output_file)

    final.to_csv(args.output_file, header=True, index=False)
    print(f"Saved to {args.output_file}")

if __name__ == '__main__':
    main()