"""Early stopping for QuantizedASR's trainer: the changes to make, and where.

Nothing here runs against QuantizedASR. This file states the diff and carries the two functions
to paste, so they can be reviewed before anyone edits that repo.

Target: `qasr/train/train_utils.py`, `build_trainer` (ends the file). Verified against
QuantizedASR at the time of writing and against `transformers` 4.57.5 in the `asr` env, whose
`EarlyStoppingCallback` is byte-identical in behaviour to the copy vendored at
`QuantizedASR/transformers/trainer_callback.py`.

WHAT IS ALREADY THERE
  Seq2SeqTrainingArguments already sets every field the callback reads: eval_strategy='steps',
  eval_steps, save_strategy='steps', save_steps, metric_for_best_model, load_best_model_at_end.
  `weight_decay=args.weight_decay` is already wired too, so weight decay needs only a CLI value
  (`--weight_decay 0.01`); it is not a code change.
  `DecoderASRTrainer.__init__` already accepts and forwards `callbacks`, so the trainer class
  does not change. Neither construction site passes `callbacks=` today -- that is the gap.

THE FOUR TRAPS, all of which silently disable or defer stopping rather than erroring:
  1. `save_steps != eval_steps` defers the stop to the next save step (documented). Both default
     to 100 here, but they are independent CLI args.
  2. `load_best_model_at_end=False` leaves `state.best_metric` unset and the callback only logs
     a warning. Note `--load_best_model_at_end` is `action='store_false'`, so the DEFAULT is
     True and passing the flag turns it off.
  3. `greater_is_better=False` is HARDCODED in build_trainer. Correct for loss/wer/cer, silently
     inverted for a higher-is-better metric such as `wip`.
  4. `prediction_loss_only=True` means compute_metrics never runs, so only `loss` is available;
     any other metric_for_best_model resolves to None and the callback disables itself with a
     warning.

Run this file to print the change list:  python add_early_stopping.py
"""

from transformers import EarlyStoppingCallback

# --- 1. qasr/misc/misc_utils.py, parse_args (beside --save_steps / --eval_steps, ~line 89) ---
NEW_CLI_ARGS = """
    parser.add_argument('--early_stopping_patience', type=int, default=0,
                        help='0 disables early stopping. N stops after N evaluations with no '
                             'improvement in metric_for_best_model.')
    parser.add_argument('--early_stopping_threshold', type=float, default=0.0,
                        help='Minimum change counted as an improvement.')
    parser.add_argument('--greater_is_better', action='store_true',
                        help='Set when metric_for_best_model improves upward (e.g. wip). '
                             'Leave unset for loss, wer and cer.')
"""

# --- 2. qasr/train/train_utils.py, top-level import (beside the existing transformers import) ---
NEW_IMPORT = "from transformers import EarlyStoppingCallback"

# --- 3. qasr/train/train_utils.py, replace the hardcoded greater_is_better in the
#        Seq2SeqTrainingArguments(...) call (currently `greater_is_better=False,`) ---
ARGS_CHANGE = "        greater_is_better=args.greater_is_better,"

# --- 4. qasr/train/train_utils.py, paste both functions above build_trainer ---------------


def validate_early_stopping(args):
    """Fail before the run starts, not 3 GPU-hours in.

    Every condition here is one the callback would otherwise meet with a warning or a silent
    deferral. Checked at construction because that is where the values are already known.
    """
    if not args.early_stopping_patience:
        return

    if args.eval_steps != args.save_steps:
        raise ValueError(
            f'early stopping needs eval_steps == save_steps, got {args.eval_steps} and '
            f'{args.save_steps}: the stop is deferred to the next save step, so the run does '
            f'not end when the metric says it should.')
    if not args.load_best_model_at_end:
        raise ValueError(
            'early stopping needs load_best_model_at_end, which is what sets state.best_metric. '
            'Note --load_best_model_at_end is store_false, so passing it is what turned this off.')
    if not args.metric_for_best_model:
        raise ValueError('early stopping needs metric_for_best_model to be set.')
    if args.prediction_loss_only and args.metric_for_best_model.replace('eval_', '') != 'loss':
        raise ValueError(
            f'prediction_loss_only=True computes no metrics, so metric_for_best_model='
            f'{args.metric_for_best_model!r} would never be found and the callback would '
            f'disable itself. Use loss, or turn prediction_loss_only off.')
    # wer and cer improve downward; greater_is_better must agree or the comparison inverts.
    lower_is_better = {'loss', 'wer', 'cer', 'mer', 'wil'}
    metric = args.metric_for_best_model.replace('eval_', '')
    if args.greater_is_better and metric in lower_is_better:
        raise ValueError(f'greater_is_better is set but {metric!r} improves downward.')


def build_callbacks(args):
    """The callback list for the trainer. Empty unless early stopping is requested."""
    if not args.early_stopping_patience:
        return None
    return [EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
    )]


# --- 5. qasr/train/train_utils.py, inside build_trainer ------------------------------------
# Call validate_early_stopping(args) immediately after `freeze_model(args, model)`, so a bad
# combination fails before the model is moved or wandb.init runs.
#
# Then pass callbacks at BOTH construction sites -- the DecoderASRTrainer branch and the
# Seq2SeqTrainer branch, since both accept it:
#
#     trainer = DecoderASRTrainer(
#         args=training_args,
#         ...
#         callbacks=build_callbacks(args),
#     )

CHANGES = [
    ('qasr/misc/misc_utils.py', 'parse_args, beside --save_steps (~line 89)',
     'add --early_stopping_patience, --early_stopping_threshold, --greater_is_better'),
    ('qasr/train/train_utils.py', 'top-level imports',
     'from transformers import EarlyStoppingCallback'),
    ('qasr/train/train_utils.py', 'Seq2SeqTrainingArguments(...)',
     'greater_is_better=False -> greater_is_better=args.greater_is_better'),
    ('qasr/train/train_utils.py', 'above build_trainer',
     'paste validate_early_stopping() and build_callbacks()'),
    ('qasr/train/train_utils.py', 'build_trainer, after freeze_model(args, model)',
     'call validate_early_stopping(args)'),
    ('qasr/train/train_utils.py', 'both trainer construction sites',
     'pass callbacks=build_callbacks(args)'),
]

# Patience is in EVALUATIONS, not steps: at eval_steps=100, patience=5 is 500 steps of no
# improvement. The runs in this study reach their best eval CER by a median of 460 steps and
# come within 1.5x of it by 170, so patience 5 would cut a 2000-step run to roughly 700-1000.
SUGGESTED = '--early_stopping_patience 5 --early_stopping_threshold 0.0 --weight_decay 0.01'

# One caveat before this goes on the comparison grid rather than on new runs: the analysis in
# LALMDecoderAnalysis reads the WHOLE curve. final_minus_best, late_sd and the eval-loss rise
# that carries the overfitting-by-tier result are all defined on a run that ran to completion.
# Early-stopped runs truncate exactly the post-minimum tail those statistics measure, so they
# belong in a new serial rather than mixed into serial 0.


def main():
    print(__doc__.split('Run this file')[0].rstrip())
    print('\nCHANGES\n')
    for path, where, what in CHANGES:
        print(f'  {path}\n    at: {where}\n    do: {what}\n')
    print(f'suggested flags: {SUGGESTED}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
