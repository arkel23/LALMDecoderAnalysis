import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import filter_df, rename_vars, drop_na, preprocess_df, sort_df, extra_columns, \
    METHODS_DIC, DATASETS_DIC, METRIC_DIC, get_canonical_labels


def make_plot(args, df):
    # Seaborn Style Settings
    sns.set_theme(
        context=args.context, style=args.style, palette=args.palette,
        font=args.font_family, font_scale=args.font_scale, rc={
            "grid.linewidth": args.bg_line_width, # mod any of matplotlib rc system
            "figure.figsize": args.fig_size,
        })
    
    if args.min_max:
        df = pd.melt(
            df,
            id_vars=['force_asr_language'],
            value_vars=['wer_norm', 'hours_norm', 'audio_norm'],
            var_name='metric',
            value_name='value'
        )

    if args.type_plot == 'bar':
        if args.hue_var_name:
            order = (
                df[df[args.hue_var_name] == df[args.hue_var_name].unique()[0]]
                .groupby(args.x_var_name)[args.y_var_name]
                .mean()
                .sort_values()  
                .index
            )
        else:
            order = (
                df.groupby(args.x_var_name)[args.y_var_name]
                .mean()
                .sort_values()  
                .index
            )
        ax = sns.barplot(x=args.x_var_name, y=args.y_var_name, hue=args.hue_var_name, data=df, order=order, errorbar=None,)
    elif args.type_plot == 'box':
        if args.x_var_name in ["Method"]:
            # print(df['Method'])
            order = (
                # df.groupby(args.x_var_name)[args.y_var_name]
                df.groupby(args.x_var_name)["Number of Parameters (10^6)"]
                .mean()
                .sort_values()  
                .index
            )
        elif args.x_var_name in ["force_asr_language", "language_resource"]:
            order = (
                # df.groupby(args.x_var_name)[args.y_var_name]
                df.groupby(args.x_var_name)["language_hours"]
                .mean()
                .sort_values()
                .index
            )
        else:
            order = (
                df.groupby(args.x_var_name)[args.y_var_name]
                .mean()
                .sort_values()  
                .index
            )
        # Dynamic, not hardcoded to a fixed set of conditions: different figures compare
        # different subsets of serials (e.g. the cross-method comparison in fig:four_plot
        # needs Full Precision/BNB8/Quanto8/HQQ4/HQQ3, while the 2-bit-specific figures need
        # Full Precision/HQQ2/HQQ3/HQQ4) -- a hardcoded list silently drops whichever
        # conditions it doesn't name (seaborn's hue_order excludes categories not listed).
        # get_canonical_labels() filters the real display order (from utils.SERIAL_DIC)
        # down to whatever's actually present in `df`, so every invocation is correct
        # automatically regardless of --keep_serials.
        hue_order = None
        if args.hue_var_name in df.columns:
            hue_order = get_canonical_labels(present=df[args.hue_var_name].unique()) or None
        
        if args.hue_var_name:
            if hue_order:
                ax = sns.boxplot(x=args.x_var_name, y=args.y_var_name, hue=args.hue_var_name, data=df, order=order, hue_order=hue_order)
            else:
                ax = sns.boxplot(x=args.x_var_name, y=args.y_var_name, hue=args.hue_var_name, data=df, order=order)
        else:
            ax = sns.boxplot(x=args.x_var_name, y=args.y_var_name, hue=args.hue_var_name, data=df, order=order)
            
    elif args.type_plot == 'violin':
        ax = sns.violinplot(x=args.x_var_name, y=args.y_var_name, hue=args.hue_var_name, data=df)
    elif args.type_plot == 'line':
        metrics = args.keep_error_metrics if args.keep_error_metrics else ["cer", "ater", "ter", "her", "coner", "ver"]

        long_df = df.melt(
            id_vars=args.x_var_name,
            value_vars=metrics,
            var_name="error_metric",
            value_name="value"
        )
        long_df["error_metric"] = long_df["error_metric"].map(METRIC_DIC).fillna(long_df["error_metric"])

        if args.x_var_name == "Status":
            # Dynamic for the same reason as the box-plot hue_order fix above: a hardcoded
            # category list silently turns any non-matching value to NaN under
            # pd.Categorical -- derive the order from utils.get_canonical_labels() (shared
            # with the box-plot branch) filtered to whatever's actually present.
            status_order = get_canonical_labels(present=long_df["Status"].unique())
            if status_order:
                long_df["Status"] = pd.Categorical(
                    long_df["Status"],
                    categories=status_order,
                    ordered=True
                )

        ax = sns.lineplot(x=args.x_var_name, y=args.y_var_name, marker=args.marker,
                          hue=args.hue_var_name, style=args.style_var_name,
                          markers=True, linewidth=args.line_width, data=long_df)
        if args.hue_var_name == 'error_metric':
            ax.legend(title='Metric')
    elif args.type_plot == 'scatter':
        ax = sns.scatterplot(x=args.x_var_name, y=args.y_var_name, hue=args.hue_var_name,
                             style=args.style_var_name, size=args.size_var_name,
                             sizes=tuple(args.sizes), legend='brief', data=df)
    elif args.type_plot == 'reg':
        ax = sns.regplot(x=args.x_var_name, y=args.y_var_name, data=df)

        if args.add_text_methods:
            for i, row in df.iterrows():
                fz = '(FZ)' if row['Status'] == 'FZ' else '(FT)'
                text =  row['Method'] + fz
                ax.text(row[args.x_var_name], row[args.y_var_name], text,
                        color='black', ha='center', va='bottom',
                        fontsize=args.font_size_methods)

        if args.add_text_correlations:
            # Compute correlations
            spearman_corr, pearson_corr, r_squared = compute_correlations(df, args.x_var_name, args.y_var_name)
            
            # Format the correlation text
            # correlation_text = (
            #     f"ρ: {spearman_corr:.2f}\n"
            #     f"r: {pearson_corr:.2f}\n"
            #     f"R²: {r_squared:.3f}"
            # )

            correlation_text = (
                f"ρ: {spearman_corr:.2f}"
            )

            # Add the correlation as a text annotation
            ax.text(
                0.05, 0.95, correlation_text, 
                transform=ax.transAxes,  # Use axes coordinates (0 to 1)
                fontsize=args.font_size_correlations, 
                verticalalignment='top', 
                horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5')
            )
    elif args.type_plot == 'heatmap':
        heatmap_data = df.pivot_table(
            index=args.y_var_name,
            columns=args.x_var_name,
            values=args.hue_var_name,
            aggfunc="mean"
        )

        ax = sns.heatmap(
            heatmap_data,
            fmt=".1f",
            annot=True,
            cbar=True
        )
    
    else:
        raise NotImplementedError

    if args.add_extra_tick:
        if args.y_var_name in ['wer', 'cer', 'wer_t']:
            ax.axhline(
                y=20,
                color='darkgreen',
                linewidth=1,
                linestyle='-',
            )

        ticks = ax.get_xticks()
        if args.x_var_name not in ['model_id', 'model_family']:
            avg_audio_length_map = (
                df.groupby(args.x_var_name)['audio_length_s_mean']
                .mean()
                .to_dict()
            )

        if args.x_var_name in ['force_asr_language']:
            hours_map = (
                df.groupby(args.x_var_name)['language_hours']
                .first()
                .to_dict()
            )
        # elif args.x_var_name in ['language_region', 'language_resource']:
        #     hours_map = (
        #         df.groupby(args.x_var_name)['language_hours']
        #         .sum()
        #         .to_dict()
        #     )
        elif args.x_var_name == 'Method':
            params_map = (
                df.groupby(args.x_var_name)['Number of Parameters (10^6)']
                .first()
                .to_dict()
            )

        for x, tick in zip(ticks, ax.get_xticklabels()):

            group = tick.get_text()

            offset = 0.1

            # Larger font -> move further down
            offset += 0.03 * (args.font_scale - 1)
            if args.x_rotation:
                # Rotated text needs more room
                offset += 0.0025 * abs(args.x_rotation)


            # if args.x_var_name not in ['Method', 'model_family']:
            #     avg_len = avg_audio_length_map.get(group)

            #     if pd.isna(avg_len):
            #         avg_len_text = "N/A"
            #     else:
            #         avg_len_minutes = avg_len / 60.0
            #         avg_len_text = f"{avg_len_minutes:.2f}m"

            #     ax.text(
            #         x,
            #         -0.075,
            #         avg_len_text,
            #         ha='center',
            #         va='top',
            #         fontsize=12 * args.font_scale * 0.75,
            #         transform=ax.get_xaxis_transform()
            #     )

            if args.x_var_name in ['force_asr_language']:

                hours = hours_map.get(group)

                if pd.isna(hours):
                    hours_text = "N/A"
                else:
                    hours_text = f"{hours:g}h"

                ax.text(
                    x,
                    -offset,
                    hours_text,
                    ha='center',
                    va='top',
                    fontsize=12 * args.font_scale * 0.75,
                    transform=ax.get_xaxis_transform()
                )
            
            if args.x_var_name in ['Method']:
                params = params_map.get(group)

                if pd.isna(params):
                    params_text = "N/A"
                elif params >= 1000:
                    params_text = f"{params/1000:.1f}B"
                else:
                    params_text = f"{params:.0f}M"

                ax.text(
                    x,
                    -offset,
                    params_text,
                    ha='center',
                    va='top',
                    fontsize=12 * args.font_scale * 0.75,
                    transform=ax.get_xaxis_transform()
                )


    
    # labels and title
    ax.set(xlabel=args.x_label, ylabel=args.y_label, title=args.title, ylim=args.y_lim)

    if args.log_scale_x:
        ax.set_xscale('log')
    if args.log_scale_y:
        ax.set_yscale('log')

    # ticks labels
    if args.x_ticks_labels:
        x_ticks = ax.get_xticks() if getattr(args, 'x_ticks', None) is None else args.x_ticks
        ax.set_xticks(x_ticks , labels=args.x_ticks_labels)

    # Rotate x-axis or y-axis ticks lables
    if (args.x_rotation != None):
        plt.xticks(rotation = args.x_rotation)
    if (args.y_rotation != None):
        plt.yticks(rotation = args.y_rotation)

    # Change location of legend
    if not args.type_plot == 'heatmap' and args.hue_var_name:
        sns.move_legend(ax, loc=args.loc_legend)

    # save plot
    output_file = os.path.join(args.results_dir, f'{args.output_file}.{args.save_format}')
    plt.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
    print('Save plot to directory ', output_file)

    return 0


def parse_args():
    parser = argparse.ArgumentParser()

    # Subset models and datasets
    parser.add_argument('--input_file', type=str,
                        default=os.path.join('data', 'qasr_400.csv'),
                        help='filename for input .csv file')

    parser.add_argument('--keep_datasets', nargs='+', type=str, default=None)
    parser.add_argument('--keep_methods', nargs='+', type=str, default=None)
    parser.add_argument('--keep_error_metrics', nargs='+', type=str, default=None,
                        help="metrics to melt/plot for --type_plot line (default: cer ater ter her coner ver)")
    parser.add_argument('--filter_datasets', nargs='+', type=str, default=None)
    parser.add_argument('--filter_methods', nargs='+', type=str, default=None)
    parser.add_argument('--keep_serials', nargs='+', type=int, default=None)
    parser.add_argument('--keep_ratios', nargs='+', type=int, default=None)
    parser.add_argument('--keep_extractor', nargs='+', type=str, default=None)

    parser.add_argument('--resource_bin', type=str, default=None)

    # Make a plot
    parser.add_argument('--log_scale_x', action='store_true')
    parser.add_argument('--log_scale_y', action='store_true')
    parser.add_argument('--type_plot', choices=['bar', 'line', 'box', 'violin', 'scatter', 'reg', 'heatmap'],
                        default='bar', help='the type of plot (line, bar)')

    parser.add_argument('--x_var_name', type=str, default='model_id',
                        help='name of the variable for x')
    parser.add_argument('--y_var_name', type=str, default='wer',
                        help='name of the variable for y')
    parser.add_argument('--hue_var_name', type=str, default=None,
                        help='legend of this bar plot')
    parser.add_argument('--style_var_name', type=str, default=None,
                        help='legend of this bar plot')
    parser.add_argument('--size_var_name', type=str, default=None,)

    parser.add_argument('--add_text_methods', action='store_true')
    parser.add_argument('--add_text_correlations', action='store_false')
    parser.add_argument('--font_size_methods', type=int, default=8)
    parser.add_argument('--font_size_correlations', type=int, default=15)

    parser.add_argument('--orient', type=str, default=None,
                        help='orientation of plot "v", "h"')

    # output
    parser.add_argument('--output_file', default='throughput_vit', type=str,
                        help='File path')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'plots'),
                        help='The directory where results will be stored')
    parser.add_argument('--save_format', choices=['pdf', 'png', 'jpg'], default='png', type=str,
                        help='Print stats on word level if use this command')

    # style related
    parser.add_argument('--context', type=str, default='notebook',
                        help='''affects font sizes and line widths
                        # notebook (def), paper (small), talk (med), poster (large)''')
    parser.add_argument('--style', type=str, default='whitegrid',
                        help='''affects plot bg color, grid and ticks
                        # whitegrid (white bg with grids), 'white', 'darkgrid', 'ticks'
                        ''')
    parser.add_argument('--palette', type=str, default='colorblind',
                        help='''
                        color palette (overwritten by color)
                        # None (def), 'pastel', 'Blues' (blue tones), 'colorblind'
                        # can create a palette that highlights based on a category
                        can create palette based on conditions
                        pal = {"versicolor": "g", "setosa": "b", "virginica":"m"}
                        pal = {species: "r" if species == "versicolor" else "b" for species in df.species.unique()}
                        ''')
    parser.add_argument('--color', type=str, default=None)
    parser.add_argument('--font_family', type=str, default='serif',
                        help='font family (sans-serif or serif)')
    parser.add_argument('--font_scale', type=float, default=1.0,
                        help='adjust the scale of the fonts')
    parser.add_argument('--bg_line_width', type=int, default=0.25,
                        help='adjust the scale of the line widths')
    parser.add_argument('--line_width', type=int, default=0.75,
                        help='adjust the scale of the line widths')
    parser.add_argument('--fig_size', nargs='+', type=float, default=[10, 6],
                        help='size of the plot')
    parser.add_argument('--sizes', type=int, nargs='+', default=[40, 1600])
    parser.add_argument('--marker', type=str, default='o',
                        help='type of marker for line plot ".", "o", "^", "x", "*"')
    parser.add_argument('--dpi', type=int, default=300)

    # Set title, labels and ticks
    parser.add_argument('--title', type=str,
                        default='Throughput of ViT Models',
                        help='title of the plot')
    parser.add_argument('--x_label', type=str, default=None,
                        help='x label of the plot')
    parser.add_argument('--y_label', type=str, default=None,
                        help='y label of the plot')
    parser.add_argument('--y_lim', nargs='*', type=int, default=None,
                        help='limits for y axis (suggest --ylim 0 100)')
    parser.add_argument('--x_ticks', nargs='+', type=int, default=None)
    parser.add_argument('--x_ticks_labels', nargs='+', type=str, default=None,
                        help='labels of x-axis ticks')
    parser.add_argument('--x_rotation', type=int, default=None,
                        help='lotation of x-axis lables')
    parser.add_argument('--y_rotation', type=int, default=None,
                        help='lotation of y-axis lables')

    # Change location of legend
    parser.add_argument('--loc_legend', type=str, default='upper right',
                        help='location of legend options are upper, lower, left right, center')
    
    # flag
    parser.add_argument('--summarized', action='store_true',
                        help='flag for making plot')
    parser.add_argument('--method_family', type=str, default = None,
                        help='choose method family to be used')
    parser.add_argument('--aggregate_dataset', action='store_true',
                        help='aggregate dataset naming variants into one')
    parser.add_argument('--add_extra_tick', action='store_true',
                        help='add extra tick for language hours')
    parser.add_argument('--min_max', action='store_true',
                        help='uses min-max values')

    args= parser.parse_args()
    return args


def process_df(args):
    df = pd.read_csv(args.input_file)

    if args.method_family == 'resnet':
        args.keep_methods = METHODS_RESNET
    elif args.method_family == 'vit':
        args.keep_methods = METHODS_VIT

    if args.aggregate_dataset:
        base_datasets = ['aircraft', 'cub', 'cars']

        df['dataset_name'] = df['dataset_name'].apply(
            lambda x: x.split('_')[0] if x.split('_')[0] in base_datasets else x
        )

    if args.summarized:
        df = extra_columns(df)
        df = filter_df(
            df,
            getattr(args, 'keep_datasets', None),
            getattr(args, 'keep_methods', None),
            getattr(args, 'keep_serials', None),
            getattr(args, 'filter_datasets', None),
            getattr(args, 'filter_methods', None),
            getattr(args, 'filter_serials', None),
            getattr(args, 'keep_ratios', None),
            getattr(args, 'keep_extractor', None),
        )
        
        # df = sort_df(df)
        # print(df['method'])
    else:
        # df['model_name'] = df['model_name'].apply(lambda x: x if str(x).startswith('hi') else f'hi{x}')
        df = preprocess_df(
            df,
            'all',
            getattr(args, 'keep_datasets', None),
            getattr(args, 'keep_methods', None),
            getattr(args, 'keep_serials', None),
            getattr(args, 'filter_datasets', None),
            getattr(args, 'filter_methods', None),
            getattr(args, 'filter_serials', None),
            getattr(args, 'keep_ratios', None),
            getattr(args, 'keep_extractor', None),
        )
        # print(df['method'].unique())
        # print(df['n_cluster_ratio'])

    # df = drop_na(df, args)

    df = rename_vars(df, var_rename=True, args=args)

    if args.resource_bin:
        df = df[df['language_resource'] == args.resource_bin]
    
    return df


def main():
    args = parse_args()
    args.title = args.title.replace("\\n", "\n")
    args.title = " ".join([METHODS_DIC.get(w, w) for w in args.title.split()])
    args.title = " ".join([DATASETS_DIC.get(w, w) for w in args.title.split()])
    os.makedirs(args.results_dir, exist_ok=True)

    if args.color:
        # single color for whole palette (sns defaults to 6 colors)
        args.palette = [args.color for _ in range(len(args.subset_models))]

    df = process_df(args)
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    # print(df[args.y_var_name])
    # print(df[args.x_var_name])

    make_plot(args, df)

    return 0

if __name__ == '__main__':
    main()