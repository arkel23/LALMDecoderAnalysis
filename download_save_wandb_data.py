import os
import argparse
import wandb
import pandas as pd


CONFIG_COLS = [
    'serial', 'dataset_path', 'dataset', 'split', 'model_id', 'quant_config', 'quant_dtype_weights',
    'force_asr_language',
]

SUMMARY_COLS = [
    'wer', 'cer', 'rtfx',
    'time_total', 'max_memory', 'flops',
    'no_params', 'num_samples', 'audio_length_s_mean'
]


SORT_COLS = [
    'serial', 'dataset_path', 'dataset', 'split', 'model_id',
    'host', 'state'
]


def get_wandb_project_runs(project, serials=None):
    api = wandb.Api()

    if serials:
        runs = api.runs(path=project, per_page=2000,
                        filters={'$or': [{'config.serial': s} for s in serials]})
    else:
        runs = api.runs(path=project, per_page=2000)

    print('Downloaded runs: ', len(runs))
    return runs


def make_df(runs, config_cols, summary_cols):
    data_list_dics = []

    for i, run in enumerate(runs):
        run_data = {}
        try:
            host = {'host': run.metadata.get('host')}
        except:
            print(run)
            host = {'host': None}
        cfg = {col: run.config.get(col, None) for col in config_cols}

        state = {'state': run.state}
        # summary = {col: run.summary.get(col, None) for col in summary_cols}
        summary_dict = dict(run.summary)

        summary = {col: summary_dict.get(col, None) for col in summary_cols}

        run_data.update(host)
        run_data.update(state)
        run_data.update(cfg)
        run_data.update(summary)

        data_list_dics.append(run_data)

        if (i + 1) % 10 == 0:
            print(f'{i}/{len(runs)}')

    df = pd.DataFrame.from_dict(data_list_dics)
    print(df.head())
    return df


def sort_save_df(df, fp, sort_cols=['serial']):
    df = df.sort_values(by=sort_cols, ascending=[True for _ in sort_cols])
    # print(df['n_cluster_ratio'])
    df.to_csv(fp, header=True, index=False)
    return 0


def parse_args():
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument('--project_name', type=str, default='eras/QASR',
                        help='project_entity/project_name')
    # Filters
    parser.add_argument('--serials', nargs='+', type=int, default=[300])
    parser.add_argument('--config_cols', nargs='+', type=str, default=CONFIG_COLS)
    parser.add_argument('--summary_cols', nargs='+', type=str, default=SUMMARY_COLS)

    # Output
    parser.add_argument('--output_file', default='qasr_400.csv', type=str,
                        help='File path')
    parser.add_argument('--results_dir', type=str, default='data',
                        help='The directory where results will be stored')
    parser.add_argument('--sort_cols', nargs='+', type=str, default=SORT_COLS)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    args.output_file = os.path.join(args.results_dir, args.output_file)

    # Get W&B runs and create the initial DataFrame
    runs = get_wandb_project_runs(args.project_name, args.serials)
    df = make_df(runs, args.config_cols, args.summary_cols)

    # Sort and save the updated DataFrame
    sort_save_df(df, args.output_file, args.sort_cols)

    return 0

if __name__ == '__main__':
    main()