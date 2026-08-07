# LALMDecoderAnalysis — what this repo is and where everything lives

Analysis-only repo for the LisTAya / TinyAya decoder-SFT comparison. It does not train or
evaluate anything; it downloads wandb runs and derives every table and figure from them.

**Start with [`docs/SUMMARY.md`](docs/SUMMARY.md)** -- every experiment and every finding on one page, findings ranked by how compelling they are for the paper.

## The research goal

**Isolate the effect of decoder SFT data composition on a LALM's listening ability.**

Everything is held constant except which Tiny Aya decoder variant is plugged in —
`tiny-aya-{base, global, earth, fire, water}`, where `base`/`global` are non-regional and
`earth`/`fire`/`water` are regional specialisations. Same Whisper-medium encoder, same
connector-only recipe (`freeze_encoder`, `freeze_decoder`, no PEFT), same eval sets.

If the variants really are identical apart from their post-training mix, differences in
downstream performance are attributable to that mix. That answers a question the LALM
literature leaves implicit: does specialising the decoder's text SFT data change what the model
can *hear*, or only what it can *say*?

## Current state

- **Serial 0 is exactly the 12 x 4 grid, 48 runs.** The control arms (`base`, `Qwen3-4B`),
  superseded re-runs and the retired `es_es` condition live in serials 1-5, so an aggregate over
  serial 0 needs no filtering. `docs/OUTPUTS.md` has the full table.
- **Serial 10** (off-the-shelf baselines) is filling in; **serial 11** (trained checkpoints) is
  not yet run.
- The headline hypothesis is a **null**, stated with a minimum detectable effect. The positive
  results are the monotone overfitting-vs-resource-tier finding and the measurement of how small
  the specialisation treatment actually is (~1.3 pp).

## Where to look

| Document | Holds |
|---|---|
| `docs/FINDINGS.md` | the evidence behind the paper |
| `docs/OUTPUTS.md` | every generated file, what writes it, how to rebuild, and the definitions |
| `docs/ROADMAP.md` | what to evaluate next, and the augmentation study (future work) |
| `docs/UPSTREAM_FIXES.md` | changes that belong in QuantizedASR, not here |
| `docs/CLAUDE_CHANGES.md` | dated chronology and rationale — the only place for narrative |
| `CLAUDE.md` | conventions for working in this repo |
| `ACL26_LALMDecoder/` | **the paper.** `main.tex` plus `tables/*.tex`, all generated |

Rebuild everything with `bash plotter.sh` in the `pytorch` env. Anything that loads a dataset
needs `asr` instead — `pytorch` has no audio backend.

`for_quantizedasr/` holds generators and sweeps destined for QuantizedASR, kept here for review;
that repo is never modified from this one.

## Scripts

| File | What it does |
|---|---|
| `download_save_wandb_data.py` | One serial's runs → `data/raw_serials/raw_serial_<N>.csv`. One serial per invocation; a combined query is much slower. |
| `download_wandb_history.py` | The per-step curves, one row per (run, step). `--history` switches it on. |
| `count_data.py` | Per-dataset run counts and `state` breakdowns. Runs before anything derives numbers. |
| `analyze_*.py` | The analysis tables t1–t9. See `docs/OUTPUTS.md` for which writes what. |
| `plot_curve.py` | Every figure. `plot.py` cannot draw training curves — see its docstring. |
| `verify_*.py`, `test_utils_port.py` | The guards; all run last in `plotter.sh` and fail loudly. |
| `rename_wandb_serial.py` | Moves runs between serials in place. Dry-run by default. |
| `missing_runs.py` | Diffs a downloaded serial against the grid its sweep would run, and writes the commands to fill the gaps. |
| `utils.py` | Data dicts plus the preprocessing pipeline. |

## Correctness risks still open

1. **The matched-variant premise is unverified.** Same base model, tokenizer and parameter count
   across the four regional variants, differing only in post-training mix — still unconfirmed
   against the Tiny Aya report. *The causal claim depends on this.* If they differ otherwise,
   reframe as a "specialisation bundle" comparison.
2. **One seed.** The minimum detectable effect is far above the effects being looked for, and the
   one replicate pair differs by 5.01 CER. This caps what any framing can claim.
3. **Task overfitting is demonstrated here, not hypothetical.** A connector trained on one fixed
   ASR prompt makes the decoder ignore other instructions. Vary instructions before extending to
   non-ASR evaluation, or the eval measures prompt-following rather than listening.
4. **Text-only and audio-only controls are required** for any MCQ work — the variants differ in
   what text knowledge they were trained on, which is exactly where that confound bites. Those
   controls do not exist upstream (`docs/ROADMAP.md` part 1, section 5).

Full design and prior analysis live in
`/mnt/c/Users/edwin.rios/Claude/Projects/LisTAya_Listening_TinyAya` — `ExecutionMasterPlan.md`
first, then `LisTAya_Publication_Assessment.md`.
