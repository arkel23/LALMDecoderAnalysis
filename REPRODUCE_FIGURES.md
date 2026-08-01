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
| `s0_volume_interaction.png` | **The headline figure.** Region-match effect against training-stream size, one point per language. |
| `s0_volume_interaction_relative.png` | The same effect as a share of baseline CER, so the volume/difficulty confound is visible rather than argued. |

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
- **symlog on the volume figures.** One language sits at -14.7 CER while the rest are within
  +/-1.2. A linear y-axis compresses six of seven points into an unreadable band; a log axis
  cannot render a signed effect at all. `--symlog_y 1.0` is linear within +/-1 and logarithmic
  outside, which is the only scaling that shows both.
- **Deterministic hue order.** `plot_curve.get_hue_order` sorts hue levels it does not find in
  `METHODS_DIC`. It previously iterated a `set`, and CPython randomises string hashing per
  process, so region colours changed between renders while point positions stayed identical --
  the exact silent-colour-shift defect the repo warns about. Renders are now byte-identical
  across processes; that is worth re-checking after any change to the function.
- **Log axes.** Both axes are log on the CER figures. Seaborn's line artists are not
  registered in the axes' data limits, so switching to log after drawing can leave a linear
  lower bound of 0 that matplotlib collapses to (~0, 1], rendering an empty panel with no
  error. `plot_curve.py` sets the limits from the positive data to avoid this.
