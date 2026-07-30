# LALMDecoderAnalysis — what this repo is for and what is in it

Created 2026-07-30. **No data has been collected here yet.** This is a fresh analysis repo,
seeded with the tooling that the sibling QASR analysis repos proved useful, so that the first
real run has somewhere to land.

## The research goal

**Isolate the effect of decoder SFT data composition on a LALM's listening ability.**

The comparison holds everything constant except the decoder's post-training data mix:

- Same audio encoder (Whisper-small), same connector recipe, same training data for the
  connector, same eval sets.
- The only thing that varies is **which Tiny Aya decoder variant** is plugged in:
  `tiny-aya-{base, global, earth, fire, water}`. `base`/`global` are non-regional; `earth`,
  `fire`, `water` are regional specialisations (the region grouping used throughout the
  LisTAya work).

If the variants really are identical apart from their SFT data mix, then differences in
downstream SLU/ASR performance are attributable to that mix — which is the claim the project
wants to make. That gives a clean answer to a question the LALM literature mostly leaves
implicit: *does specialising the decoder's text SFT data actually change what the model can
hear, or only what it can say?*

**Full design, motivation and prior analysis live in
`/mnt/c/Users/edwin.rios/Claude/Projects/LisTAya_Listening_TinyAya`.** Start with
`ExecutionMasterPlan.md` (phased plan, and an inventory of what the upstream codebase already
supports), then `LisTAya_Publication_Assessment.md` (framing and correctness risks) and
`Chat_Survey_LALMDecoderSFTEvalFeasibility.md`. There is also a slide deck
(`20260518_LALMDecoderLowResourceSLU.pptx`), two bib files, and per-person work under
`LisTAyaDrive/`.

## Where the experiments actually run

This repo does **analysis only** — it does not train or evaluate anything, exactly as
MultilingualQASR relates to the upstream eval codebase.

- Models are built by `qasr/model/custom_q2a.py:build_q2a()` in **QuantizedASR-main**, which
  already accepts any Whisper encoder + any causal-LM decoder and documents
  `tiny-aya-{base,global,earth,fire,water}` as supported `lm_version` values.
- Five ready-made configs already exist there:
  `configs/models/cq2a_whisper_small_tiny_aya_{base,global,earth,fire,water}.yaml`.
- `tools/train.py` supports `--freeze_encoder --freeze_decoder`, i.e. the SLAM-style
  connector-only recipe this design calls for.
- SLU eval goes through `qasr/eval/slu.py` with `task: qa_mcq`, which already routes
  correctly for `q2a_`-prefixed model IDs.

Per the master plan, the infrastructure largely exists and the project's problem has been
that work was happening in per-person notebooks instead of on that codebase.

## Tools available in this repo

| File | What it does |
|---|---|
| `download_save_wandb_data.py` | Pulls one serial's runs from a wandb project into `data/raw_serials/raw_serial_<N>.csv`. **Download one serial per invocation** — a combined `--serials` query is much slower. |
| `concat_df.py` | Row-wise concatenation of two per-serial CSVs. Chain it to build the combined table. |
| `count_data.py` | Sanity-checks a downloaded CSV: per-serial and per-dataset run counts, and `state` breakdowns. Run it before trusting new data. |
| `plot.py` | The plotting CLI (box/line/scatter over `x_var_name`/`y_var_name`/`hue_var_name`, with filtering by dataset/method/serial). Writes into `results_all/plots/`. |
| `utils.py` | Data dicts plus the preprocessing pipeline (`preprocess_df` → standardise → keep columns → derive → filter → sort). |
| `.gitignore` | Inherited; already excludes checkpoints, logs, `__pycache__`, wandb caches. |
| `CLAUDE.md` | Conventions, pruned from the shared analysis-repo file. |

`plot.py` imports from `utils.py` and both are verified to import cleanly, but see the
warning below.

### Read this before using `utils.py`

Its dicts are **placeholders describing the intended matrix, not observed data**:

- `SERIAL_DIC` is **empty**. Until you populate it, `get_canonical_labels()` returns `[]` and
  any figure relying on hue ordering will be wrong or empty.
- `METHODS_DIC` / `MODEL_FAMILY` list the five TinyAya variants with a Regional /
  Non-regional grouping — confirm the model_id strings match what wandb actually logs.
- `DATASETS_DIC` is empty; fill it to control dataset ordering in figures.
- The multilingual machinery from MultilingualQASR (language-hours table, resource tiers,
  the `NEEDS_CER` primary-error-rate switch) was **deliberately removed**. If per-language
  scoring is needed, add it deliberately.

`get_canonical_labels` de-duplicates on purpose: a `hue_order` containing a duplicate
silently mis-assigns one category's colour in seaborn while positions and labels stay
pixel-identical. That was a real bug in a sibling repo.

## Correctness risks to settle early

These come from the project's own assessment documents and are worth stating here because
they determine whether the headline claim is defensible at all.

1. **Verify the Tiny Aya variants are genuinely matched** — same base model, tokenizer and
   parameter count, differing only in regional post-training — against the official Tiny Aya
   report. *The entire causal claim depends on this.* If they differ in other ways, reframe as
   a "specialisation bundle" comparison rather than a data-mix comparison.
2. **Task overfitting is a demonstrated failure mode here, not a hypothetical.** A connector
   trained on one fixed ASR prompt makes the decoder ignore other instructions (SALMONN's
   "task overfitting"); the project already produced an instance of this in a notebook that
   trained on a single `"Transcribe the speech: <audio>"` prompt. Vary instructions during
   connector training before running the full matrix, or the eval measures prompt-following
   rather than listening.
3. **Text-only and audio-only MCQ controls are required**, not optional. Without them a high
   MCQ score may reflect the decoder answering from prior knowledge instead of from the audio
   — and a decoder-variant comparison is exactly the setting where that confound bites, since
   the variants differ in what text knowledge they were trained on.
4. **Known upstream blocker**: WaxalNLP's `dag` (Dagbani) config raises a `CastError`. Avoid
   that pair or fix the loader first.
5. `sneh/create csv/dataset_registry.csv` (1,743 rows) and the YAML configs are currently two
   disconnected systems. The registry should drive config generation rather than being
   re-derived by hand.

## Suggested first steps

1. Settle risk 1 above — it is cheap and it gates the framing.
2. Get one serial of eval runs logged to wandb for two variants, download it with
   `download_save_wandb_data.py`, and check it with `count_data.py`.
3. Populate `SERIAL_DIC` / `METHODS_DIC` / `DATASETS_DIC` from that real data.
4. Create `plotter.sh` at that point, not later — one command from a bare checkout should
   download, concatenate, analyse and plot. Retrofitting this was expensive in the sibling
   repos.
5. Add `verify_paper_numbers.py` and unit tests alongside the *first* table rather than after
   the paper exists. In MultilingualQASR that harness immediately found five wrong printed
   numbers and two double-rounding defects.

## Status

Nothing committed to `data/`, `results_all/` or `configs/` yet — those directories are empty
placeholders. There is no `plotter.sh`, no paper, and no tests, by design: this repo is a
seeded starting point, not work in progress.
