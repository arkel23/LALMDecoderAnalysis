# Roadmap: what to evaluate next, and what to try after that

Merges the evaluation-dataset plan (2026-07-30) and the augmentation study (2026-08-03).
ASR-first by decision: non-ASR evaluation is explicitly lower priority, recorded here so it is
not rediscovered, not scheduled. Literature numbers cited below are sourced in `RELATED_WORK.md`.

---

# Part 1 — Evaluation

## The problem this fixes

Ten of the twelve languages **train on WorldSpeech and evaluate on FLEURS**. WorldSpeech is
parliamentary proceedings, international broadcasts and public-domain audiobooks; FLEURS is read
speech. So most reported numbers are domain-transfer numbers, and the grid cannot separate "this
decoder specialises better" from "this decoder transfers across domain better". Only `ha_ng`
(WorldSpeech test) and `crs_sc` (ERISLab val_clean) are in-domain, so cross-language comparisons
also mix two different quantities.

## 1. Same-language in-distribution — DONE, not yet run

`create_yamls_worldspeech_lalm.py` now generates all 120 WorldSpeech `test` configs, so every
language has an in-domain point to sit beside its FLEURS point. `eval_lalm_decoder_txf.sh` runs
them under serial 11: the **best** checkpoint of each of the four grid variants against all 44
datasets of its own languages, so 176 evaluations. Step 1000 is not evaluated -- it is the
overfitted end of the curve, and the best-to-final gap already comes off the training curves.
`--models fire` restricts the sweep to one variant's 44.

The decode bug that blocked this was an environment problem, not a corpus one: the `pytorch` env
has no audio backend at all (`datasets` 5.0.0, neither soundfile nor torchcodec). In `asr`,
`la_va` / `si_lk` / `tl_ph` all decode correctly via torchcodec at 24 kHz. **Use `asr` for
anything that loads audio.**

Run `verify_dataset_durations.py --load` on a newly introduced split before trusting it — the
`duration`-column inconsistency that forced the `crs_sc` cleaning pass could affect others.

## 2. Same-language, different domain — reframe what exists

Keep FLEURS as the **deliberate** held-out domain rather than the accidental primary. 229 configs
already exist in QuantizedASR, so this costs nothing beyond re-labelling.

The current runs evaluate on FLEURS **validation**. Switching the headline to FLEURS `test`
avoids reporting on a split that was implicitly model-selected against, since best-CER is chosen
over the eval curve.

**`ha_ng` is the exception, and switching splits does not fix it.** Hausa selected on
`disco-eth/WorldSpeech ha_ng test` -- the same config the sweeps use as its in-domain point --
so that number is optimistically biased where the other eleven are not. WorldSpeech has only a
95/5 train/test split, so there is no untouched in-domain split for `ha_ng`. The analysis
therefore uses **`ha_td`** as Hausa's in-domain point (`utils.IN_DOMAIN_PRIMARY`): it was in
Hausa's training mix, so it is equally in-domain, and it was never used for selection.
`ha_ng` is still evaluated, and is reported as accent transfer.

## 3. Cross-language and cross-region transfer

Evaluate each trained connector zero-shot on the other language(s) in its own region, and one
language per other region. Eval-only, uses existing `fleurs_*` configs. This is the only tier
that supports *region*-level rather than *language*-level wording — two languages per region is
thin for family-level claims, and evaluation is cheap.

## 4. Out-of-distribution and accent

- **The held-out WorldSpeech variants are now the cheapest accent axis** and need no new
  configs: 7 English, 8 Spanish, 2 French varieties never in the training mix. `--eval_set all`
  runs them. This is what EdAcc was previously wanted for, available in-domain.
- **EdAcc (`edinburghcstr/edacc`)** — still no config; a one-file addition on the
  `configs/datasets/short_en/*.yaml` schema, 24 accent slices, with Qwen2-Audio-7B /
  Voxtral-Mini-3B toplines already run by the team.
- **`crs_sc`** is the strongest OOD probe in the grid: unsupported by both Whisper and TinyAya.
  Keep it framed as the encoder-and-decoder-unseen quadrant, not an ordinary language.
- **`ta_lk`** is a Tamil variety outside the training mix, so it is held-out dialect transfer --
  the same axis as the English and Spanish variants.
- **Long-form** if wanted: `configs/datasets/long_ml/` has 96 `espnet/floras` configs.

## 5. Lower-priority backlog — recorded, not scheduled

- **Belebele-FLEURS MCQ** — only `eng_Latn` wired; the loader is language-generic. Two blockers:
  `qasr/eval/slu.py` supports `{sentiment, qa_mcq, qa_generative}` but **text-only / audio-only
  controls do not exist** (every prompt builder hardcodes the audio block), and there is no
  scored SLU runner. `belebele_fleurs_utils.py` exposes a `strategy ∈ {best, worst, random}`
  audio-selection knob — a ready-made difficulty dial.
- **MELD sentiment** — config exists.
- **llama-questions** — config exists but has no `task:` field, so it runs as plain ASR/WER.
- **Speech-MASSIVE**, **IndicVoices / Kathbath**, **WaxalNLP** — no configs.

## Recommended order

1. Run serial 11 on both domains (configs and sweep exist).
2. Zero-shot cross-language sweep on FLEURS.
3. `--eval_set all` for the accent axis.
4. Add EdAcc.

All are evaluation-only and need no retraining.

---

# Part 2 — Can augmentation substitute for training hours?

**Status: NOT current work.** Nothing here is scheduled or built.

## Why it is worth doing later

SLAM-ASR needs **100–200 h** to match a Whisper-only baseline, and **five of our twelve cells sit
below it**. The nearest prior work uses no augmentation at all despite naming data scarcity as
the bottleneck, so "can augmentation substitute for hours" is open and unaddressed.

A second motivation is in this project's own record: SALMONN-style task overfitting is a
demonstrated failure mode here — a connector trained on one fixed ASR prompt stops following
other instructions. Arm B attacks that and data scarcity with the same intervention.

## The three arms

| arm | audio | target | feasible with frozen decoder? | build cost |
|---|---|---|---|---|
| **A. Audio augmentation** | perturbed (RIR / noise / speed) | unchanged transcript | yes | new module |
| **B. Target augmentation** | unchanged | synthesised tasks | yes | **already exists upstream** |
| **C. Both** | perturbed | synthesised tasks | yes | A + B |

**A** adds acoustic variation over the same linguistic content; **B** adds linguistic and task
variation over the same acoustics. If only A helps, the bottleneck is acoustic coverage; if only
B helps, it is supervision density per audio hour. That distinction is the research question, and
it is why one arm alone would be uninformative.

### Arm B already exists

`tools/preprocess/augment_lalm_sft_data.py` (QuantizedASR) uses a text-LLM teacher over each
transcript to synthesise 15 task types, adding `instruction` / `response` columns with audio and
transcript untouched, and keeps `asr_transcribe` as a low-weight (0.5) anchor.

Wiring already present upstream: `qasr/data/data_utils.py:184-192` swaps the target to `response`
when an `instruction` column exists, and `qasr/data/collators.py:404-410` builds the instruction
prompt. Our runs log `model_type: qwen2_audio`, so `QwenAudioCollator` applies and both hooks
work. Needs a column mapping — it reads LibriSpeech-style `text` while WorldSpeech uses
`human_transcript`.

Keeping the audio and changing only the **target** backpropagates into the connector with encoder
and decoder frozen, because the forward pass still runs audio → connector → decoder and the loss
lands on the new response tokens.

### Arm A does not exist at all

No SpecAugment, noise mixing, RIR convolution or speed perturbation anywhere outside the vendored
`transformers/` copy. SpecAugment is not implemented in the Qwen2-Audio encoder path, so flipping
a config flag would be a no-op.

**Insertion point:** `BaseASRCollator._extract_audios` (`qasr/data/collators.py:140`) — the single
funnel every waveform passes as 16 kHz float32 before the feature extractor. Train/eval separation
is already free (two collator objects, selected at `qasr/train/train_utils.py:139`), randomness is
fresh per batch, and DSP parallelises in dataloader workers.

**Not `prepare_data`**: it takes no `train` argument and the local-dataset path loads eagerly, so
a `.map()` transform would be cached into one frozen realisation.

## Blockers and confounds

- **`WhisperCollator` has no `inference` field**, so the train/eval gate is not leak-proof for a
  pure-Whisper run until one is added.
- **Speed perturbation changes clip duration** and interacts with the strict `< 30 s` cap — a
  1.1× slow-down can push a 27 s clip past it. That is exactly the failure mode that cost 72 % of
  the Tamil training data.
- **Teacher-quality confound in arm B**: the teacher is a *text* LLM whose coverage of Amharic,
  Hausa and Urdu is weaker than of English, so teacher quality co-varies with the axis under
  study. Needs an in-language versus English-target control.
- **Licensing**: `treble-technologies/Treble10-RIR` is CC-BY-NC-SA-4.0 — non-commercial and
  share-alike, unlike Treble10-Speech's CC-BY-4.0. Derived data inherits share-alike.

## A reference arm would be required

Cross-lingual projector pretraining is large and proven (`RELATED_WORK.md` §4). Without such an
arm, a null on augmentation cannot be distinguished from "nothing would have helped at this data
budget" — the same trap the region-match null fell into before the exposure magnitude was
measured.

## Not an arm

True text-only adaptation, with no audio at all, requires LoRA inside the decoder. That would
unfreeze the decoder and break the "only the decoder's SFT mix differs" control this study rests
on, so results would no longer be comparable across arms.

## If it is ever scheduled

The three lowest cells (`ta_in`, `am_et`, `ur_pk`) plus one high-resource control (`id_id`), 4
variants each, arms {baseline, A, B, C, cross-lingual pretraining}. Advance prediction worth
committing to: if augmentation substitutes for hours, a ~38 h cell under arm C should approach the
100–200 h band. If it does not, that bounds augmentation as a substitute for data — itself
publishable, given the gap in the closest prior work.
