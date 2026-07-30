"""
Training-curve plotting: one continuous variable against another, with a hue and optional
per-language facets.

Why this is not a branch inside plot.py. plot.py's 'line' branch cannot draw this, for
three separate reasons, none of which is a small fix:

  1. It force-melts the frame onto a hardcoded metric list (plot.py:93-101), so
     --y_var_name is not read as a data column at all -- every caller is pushed into
     y='value', hue='error_metric'.
  2. utils.keep_columns whitelists columns, so '_step', 'audio_hours' and 'train/loss'
     are silently dropped before plotting.
  3. rename_vars rewrites the axis arguments in place via VAR_DIC.

Editing that branch would change every existing figure in the sibling repos that depends on
it. So this is a separate entry point that reads the history CSV raw, and it deliberately
reuses everything reusable: the same seaborn theme block, the same style/context/palette/
font/figsize/dpi/save-format arguments, the same results_dir + output_file + extension save
convention, and utils.filter_df / METHODS_DIC / DATASETS_DIC / LANGUAGE_DIC for labelling.

Three things it adds that plot.py lacks:
  --facet_var_name   per-language grid via seaborn relplot
  --errorbar         EXPLICIT. seaborn's lineplot default is ('ci', 95) with bootstrapping,
                     which silently draws a confidence band whenever several rows share an
                     x value. Inheriting that by accident is how a figure ends up showing an
                     interval nobody chose.
  mkdir -p on the output subdirectory, which plot.py does not do (it only creates
  --results_dir), so a subdirectory in --output_file fails on a bare checkout.

Usage:
    python plot_curve.py --input_file data/raw_serials/history_serial_0.csv \
        --x_var_name audio_hours --y_var_name 'eval/cer' --hue_var_name model_id \
        --facet_var_name dataset --log_scale_x --output_file s0/s0_curve_cer_vs_audio_hours
"""
import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import filter_df, METHODS_DIC, DATASETS_DIC, LANGUAGE_DIC, METRIC_DIC, VAR_DIC


# Axis labels for the columns this script plots, which are not in utils.VAR_DIC because
# they only exist in the history CSV.
CURVE_VAR_DIC = {
    'audio_hours': 'Audio processed (hours)',
    'train/train_audio_seconds': 'Audio processed (s)',
    'eval/cer': 'Eval CER (%)',
    'eval/loss': 'Eval loss',
    'train/loss': 'Train loss',
    'train/grad_norm': 'Gradient norm',
    'train/learning_rate': 'Learning rate',
    'train/num_input_tokens_seen': 'Input tokens seen',
    '_step': 'Logged step',
    'train/global_step': 'Training step',
    'train/epoch': 'Epoch (wandb estimate; unreliable under streaming)',
    'dataset': 'Language',
    'model_id': 'Decoder variant',
}


def label_for(name):
    for d in (CURVE_VAR_DIC, VAR_DIC, METRIC_DIC):
        if name in d:
            return d[name]
    return name


def get_hue_order(present):
    """Legend order from METHODS_DIC, restricted to what is actually present.

    De-duplicated on purpose. A hue_order containing a duplicate silently corrupts
    seaborn's palette assignment -- one category gets drawn in another's colour while line
    positions and labels stay pixel-identical, so a content-only check does not catch it.
    Restricting to what is present matters for the same reason in reverse: a hue_order
    naming a category seaborn never sees shifts every subsequent colour.
    """
    present = set(present)
    seen, out = set(), []
    for label in METHODS_DIC.values():
        if label in present and label not in seen:
            seen.add(label)
            out.append(label)
    for label in present:                      # anything unmapped, stable order
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def process_df(args):
    df = pd.read_csv(args.input_file)

    if args.keep_states:
        df = df[df['state'].isin(args.keep_states)]

    df = filter_df(
        df,
        keep_datasets=None, keep_methods=args.keep_methods,
        keep_serials=args.keep_serials,
        filter_datasets=None, filter_methods=args.filter_methods,
        filter_serials=args.filter_serials,
    )
    # filter_df keys datasets on 'dataset_name', which the history CSV does not build --
    # language filtering here is on the raw language code instead.
    if args.keep_languages:
        df = df[df['dataset'].isin(args.keep_languages)]
    if args.filter_languages:
        df = df[~df['dataset'].isin(args.filter_languages)]

    # Rows where y is null are the interleaved train-log rows (for an eval metric) or the
    # eval rows (for a train metric). Dropping them is what makes the line continuous
    # instead of dashed-with-gaps.
    df = df.dropna(subset=[args.y_var_name, args.x_var_name])

    if args.log_scale_x:
        df = df[df[args.x_var_name] > 0]
    if args.log_scale_y:
        df = df[df[args.y_var_name] > 0]

    if args.hue_var_name == 'model_id':
        df['model_id'] = df['model_id'].map(lambda v: METHODS_DIC.get(v, v))
    if args.facet_var_name == 'dataset' or args.hue_var_name == 'dataset':
        df['dataset'] = df['dataset'].map(lambda v: LANGUAGE_DIC.get(v, v))
    if 'dataset_name' in df.columns:
        df['dataset_name'] = df['dataset_name'].map(lambda v: DATASETS_DIC.get(v, v))

    if df.empty:
        raise SystemExit('No rows left to plot after filtering.')
    return df


def make_plot(args, df):
    # Same theme block as plot.py, so figures from the two scripts sit side by side.
    sns.set_theme(
        context=args.context, style=args.style, palette=args.palette,
        font=args.font_family, font_scale=args.font_scale, rc={
            "grid.linewidth": args.bg_line_width,
            "figure.figsize": args.fig_size,
        })

    hue_order = None
    if args.hue_var_name:
        hue_order = get_hue_order(df[args.hue_var_name].unique())

    common = dict(
        x=args.x_var_name, y=args.y_var_name, hue=args.hue_var_name,
        hue_order=hue_order, style=args.style_var_name,
        linewidth=args.line_width, errorbar=args.errorbar, data=df,
    )
    if args.marker:
        common.update(marker=args.marker, markers=True)

    if args.facet_var_name:
        grid = sns.relplot(
            kind='line', col=args.facet_var_name, col_wrap=args.col_wrap,
            facet_kws={'sharey': args.share_y, 'sharex': args.share_x},
            height=args.facet_height, aspect=args.facet_aspect, **common)
        axes = grid.axes.flat
        grid.set_titles('{col_name}')
        grid.set_axis_labels(args.x_label or label_for(args.x_var_name),
                             args.y_label or label_for(args.y_var_name))
        if grid.legend is not None:
            grid.legend.set_title(label_for(args.hue_var_name))
        fig = grid.figure
    else:
        ax = sns.lineplot(**common)
        ax.set_xlabel(args.x_label or label_for(args.x_var_name))
        ax.set_ylabel(args.y_label or label_for(args.y_var_name))
        if args.hue_var_name:
            ax.legend(title=label_for(args.hue_var_name), loc=args.loc_legend)
        axes = [ax]
        fig = ax.figure

    for ax in axes:
        if args.log_scale_x:
            ax.set_xscale('log')
        if args.log_scale_y:
            ax.set_yscale('log')
        # Seaborn's line artists are not registered in the axes' data limits, so switching
        # to a log scale after drawing leaves the linear lower bound of 0, which matplotlib
        # collapses to (~0, 1] -- rendering an empty panel with no error. Set the limits
        # from the positive data instead.
        if args.log_scale_y and args.y_lim is None:
            yv = pd.to_numeric(df[args.y_var_name], errors='coerce')
            yv = yv[yv > 0]
            if len(yv):
                ax.set_ylim(yv.min() / 1.5, yv.max() * 1.5)
        if args.y_lim:
            ax.set_ylim(*args.y_lim)

    if args.title:
        fig.suptitle(args.title.replace('\\n', '\n'), y=1.02)

    return fig


def parse_args():
    p = argparse.ArgumentParser()

    # Input / filtering
    p.add_argument('--input_file', type=str,
                   default=os.path.join('data', 'raw_serials', 'history_serial_0.csv'))
    p.add_argument('--keep_methods', nargs='+', type=str, default=None)
    p.add_argument('--filter_methods', nargs='+', type=str, default=None)
    p.add_argument('--keep_languages', nargs='+', type=str, default=None)
    p.add_argument('--filter_languages', nargs='+', type=str, default=None)
    p.add_argument('--keep_serials', nargs='+', type=int, default=None)
    p.add_argument('--filter_serials', nargs='+', type=int, default=None)
    p.add_argument('--keep_states', nargs='+', type=str, default=None,
                   help="e.g. finished -- running runs have truncated curves")

    # Variables
    p.add_argument('--x_var_name', type=str, default='audio_hours')
    p.add_argument('--y_var_name', type=str, default='eval/cer')
    p.add_argument('--hue_var_name', type=str, default='model_id')
    p.add_argument('--style_var_name', type=str, default=None)
    p.add_argument('--facet_var_name', type=str, default=None)
    p.add_argument('--col_wrap', type=int, default=5)
    p.add_argument('--facet_height', type=float, default=2.6)
    p.add_argument('--facet_aspect', type=float, default=1.1)
    p.add_argument('--share_y', action='store_true')
    p.add_argument('--share_x', action='store_true')

    # Statistics -- explicit, never inherited
    p.add_argument('--errorbar', type=str, default=None,
                   help="seaborn errorbar spec: 'sd', 'se', 'ci' or None (default). "
                        "Left unset there is one line per hue level per x, so no band.")

    # Axes
    p.add_argument('--log_scale_x', action='store_true')
    p.add_argument('--log_scale_y', action='store_true')
    p.add_argument('--y_lim', nargs='*', type=float, default=None)

    # Style -- same names and defaults as plot.py
    p.add_argument('--context', type=str, default='notebook')
    p.add_argument('--style', type=str, default='whitegrid')
    p.add_argument('--palette', type=str, default='colorblind')
    p.add_argument('--font_family', type=str, default='serif')
    p.add_argument('--font_scale', type=float, default=1.0)
    p.add_argument('--bg_line_width', type=float, default=0.5)
    p.add_argument('--line_width', type=float, default=1.6)
    p.add_argument('--fig_size', nargs='+', type=float, default=[10, 6])
    p.add_argument('--marker', type=str, default=None)
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--title', type=str, default=None)
    p.add_argument('--x_label', type=str, default=None)
    p.add_argument('--y_label', type=str, default=None)
    p.add_argument('--loc_legend', type=str, default='upper right')

    # Output
    p.add_argument('--output_file', type=str, default='s0/s0_curve')
    p.add_argument('--results_dir', type=str, default=os.path.join('results_all', 'plots'))
    p.add_argument('--save_format', type=str, default='png',
                   choices=['pdf', 'png', 'jpg'])

    return p.parse_args()


def main():
    args = parse_args()
    df = process_df(args)
    fig = make_plot(args, df)

    output_file = os.path.join(args.results_dir, f'{args.output_file}.{args.save_format}')
    # plot.py only creates --results_dir, so a subdirectory inside --output_file raises
    # FileNotFoundError on a bare checkout. Create the full path here.
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    fig.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print('Save plot to directory ', output_file)
    return 0


if __name__ == '__main__':
    main()
