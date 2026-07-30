"""
Concatenates two results CSVs (e.g. two per-serial downloads from
download_save_wandb_data.py) row-wise into a new CSV. Intended to be chained:
run once per additional serial to build up a combined file without ever
re-downloading multiple serials in a single slow wandb API call.

Usage:
    python concat_df.py --input_file1 data/2xx/qasr_200.csv --input_file2 data/2xx/qasr_205.csv --output_file data/2xx/temp1.csv
    python concat_df.py --input_file1 data/2xx/temp1.csv --input_file2 data/2xx/qasr_206.csv --output_file data/2xx/temp2.csv
    ...
"""
import argparse
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_file1', type=str, required=True)
    p.add_argument('--input_file2', type=str, required=True)
    p.add_argument('--output_file', type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    df1 = pd.read_csv(args.input_file1)
    df2 = pd.read_csv(args.input_file2)
    df = pd.concat([df1, df2], axis=0, ignore_index=True)
    df.to_csv(args.output_file, index=False)
    print(f'Concatenated {len(df1)} + {len(df2)} = {len(df)} rows -> {args.output_file}')


if __name__ == '__main__':
    main()
