"""Concatenates two results CSVs row-wise. Chain it once per additional serial to build a
combined file without a single slow multi-serial wandb call.

Usage:
    python concat_df.py --input_file1 a.csv --input_file2 b.csv --output_file temp1.csv
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
