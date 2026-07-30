# Reproducing the figures

    conda activate pytorch
    bash plotter.sh

That rebuilds everything from a bare checkout: both wandb downloads, every analysis CSV,
every figure, and the verification guards. The downloads are guarded on file existence, so
re-running does not re-download; delete the file under `data/raw_serials/` to force a refresh.

All figures are written by `plot_curve.py` into `results_all/plots/s0/`.

| Figure | Content |
|---|---|
| `s0_curve_cer_vs_audio_hours.png` | Eval CER vs audio processed, hue = decoder variant, faceted by language. The headline figure. |
| `s0_curve_evalloss_vs_audio_hours.png` | Eval loss on the same axes — separates "still learning" from "CER is noisy". |
| `s0_curve_trainloss_vs_audio_hours.png` | Training loss on the same axes. |
| `s0_curve_crs_ood.png` | The `crs_sc` cell alone: the language unseen by both encoder and decoder. |

## Why `plot_curve.py` rather than `plot.py`

`plot.py`'s `line` branch cannot draw a training curve. It force-melts the frame onto a
hardcoded metric list (`plot.py:93-101`), so `--y_var_name` is never read as a data column;
`utils.keep_columns` whitelists columns, so `audio_hours` and `train/loss` are dropped before
plotting; and `rename_vars` rewrites the axis arguments. Fixing that branch would change
every existing figure in the sibling repos that depends on it, so `plot_curve.py` is a
separate entry point that reads the history CSV raw while reusing the same seaborn theme,
the same style/context/palette/font/figsize/dpi arguments, and the same save convention.

## Things to check by eye after regenerating

- **Colours, not just positions and labels.** A `hue_order` containing a duplicate silently
  mis-assigns one category's colour while line positions and legend text stay pixel-identical,
  so a content-only check does not catch it. `plot_curve.get_hue_order` de-duplicates and
  restricts to categories actually present, which also keeps colours consistent **across**
  figures — `TinyAya-base` appears only in the `crs_sc` panel, and every other variant must
  keep the same colour there as in the faceted figure.
- **Facet count.** The faceted figure shows **9** languages, not 10: all three `es_419` runs
  are still training and `--keep_states finished` drops them. When they finish, the figure
  gains a panel.
- **Captions against declared filters.** `s0_curve_crs_ood.png` says *5 finished TinyAya
  variants*, because the sixth model (the Qwen3-4B control) is still training and is filtered
  out. Update the title in `plotter.sh` when it finishes — a caption describing data the
  declared filter removed is a defect that shipped in a sibling repo.
- **Log axes.** Both axes are log on the CER figures. Seaborn's line artists are not
  registered in the axes' data limits, so switching to log after drawing can leave a linear
  lower bound of 0 that matplotlib collapses to (~0, 1], rendering an empty panel with no
  error. `plot_curve.py` sets the limits from the positive data to avoid this.
