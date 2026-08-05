# Deprecated

## `eval_lalm_decoder.sh` (moved 2026-08-03)

Swept the *untrained* compositions (stock encoder + stock decoder + randomly-initialised
connector) as a full model x dataset cross product, under serial 420. Superseded by
`for_quantizedasr/scripts/eval_lalm_decoder_txf.sh`, which evaluates the trained checkpoints
under serial 11 and pairs each one only with its own language.

Restore by moving it back to `for_quantizedasr/scripts/`; it needs the untrained
`cq2a_whisper_medium_tiny_aya_*.yaml` model configs, which QuantizedASR already ships.
