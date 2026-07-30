# Evaluation-dataset plan

Written 2026-07-30. ASR-first by decision: non-ASR evaluation (MCQ / sentiment /
instruction-following) is explicitly **lower priority** and is recorded in section 5 so it is
not rediscovered, not scheduled.

Config availability below was checked against `/home/edwinrios/projects/QuantizedASR`
(`configs/datasets/`, 416 YAMLs across 8 subdirectories; there is no `configs/eval/`).

---

## The problem this plan exists to fix

Eight of the ten executed languages **train on WorldSpeech and evaluate on FLEURS**:

| language | train | eval |
|---|---|---|
| en_us, es_419, fr_fr, hi_in, id_id, mr_in, sw_ke, ta_in | `disco-eth/WorldSpeech` | `google/fleurs` validation |
| ha_ng | `disco-eth/WorldSpeech` | `disco-eth/WorldSpeech` test |
| crs_sc | `ERISLab/WorldSpeech` | `ERISLab/WorldSpeech` val_clean |

WorldSpeech is parliamentary proceedings, international broadcasts and public-domain
audiobooks. FLEURS is read speech. So for 8 of 10 languages **every reported number is a
domain-transfer number**, and the grid cannot separate "this decoder specialises better" from
"this decoder transfers across domain better". That is a confound sitting underneath the
headline contrast, not a refinement.

It is also not uniform: `ha_ng` and `crs_sc` are in-domain, the other eight are not, so
cross-language comparisons mix two different quantities.

## 1. Same-language in-distribution — highest priority

**Add WorldSpeech `test` eval configs for the eight FLEURS-evaluated languages.** This is the
single change that makes the existing numbers interpretable, because it gives every language
an in-domain point to sit beside its FLEURS point.

Cost is not zero, and should be budgeted honestly:

- Only three WorldSpeech eval configs exist today —
  `configs/datasets/short_ml/worldspeech_{la_va,si_lk,tl_ph}_test.yaml` — and **all three fail
  to decode in this environment**: `examples/explore_datasets.py:30` records a real
  libsndfile/Opus error (`Supported file format but file is malformed`). That is an
  environment/dependency limitation rather than a data problem, but it blocks this tier until
  resolved.
- WorldSpeech defines a 95/5 train/test split per country-language pair, so the test split
  exists for every trained config.
- The `duration`-column inconsistency that forced the `crs_sc` cleaning pass may affect other
  configs -- `ta_in` shows exactly that signature (`PLAN_ASSESSMENT.md` §4.1). Run the
  duration-consistency check before trusting a new split.

## 2. Same-language, different domain — already in hand, needs reframing

Keep FLEURS as the **deliberate** held-out domain rather than the accidental primary. 204
configs already exist (`configs/datasets/short_ml/fleurs_<lang>_{dev,test}.yaml`, 102
languages × dev/test), so this tier costs nothing beyond re-labelling what is already run.

Note the current runs evaluate on FLEURS **validation**. Switching the headline to FLEURS
`test` avoids reporting on a split that was implicitly model-selected against, since best-CER
is chosen over the eval curve.

## 3. Cross-language and cross-region transfer — cheap, and it is what supports region claims

Evaluate each trained connector zero-shot on:

- the other language(s) in its own region (e.g. the `ha_ng` connector on `sw_ke`), and
- one language per other region.

This is eval-only, uses the existing `fleurs_*` configs and the existing sweep runner
(`scripts/eval_short_ml_v2.sh`), and it is the only tier that supports *region*-level rather
than *language*-level wording. The publication assessment already flagged that two languages
per region is thin for family-level claims and that evaluation is cheap — this is the remedy
it proposed.

## 4. Out-of-distribution and accent

- **EdAcc (`edinburghcstr/edacc`) — no config exists.** It is a one-file addition on the
  existing `configs/datasets/short_en/*.yaml` schema. It is the cheapest genuine accent axis,
  it has 24 accent slices, and Qwen2-Audio-7B / Voxtral-Mini-3B toplines have already been run
  by the team (`LisTAyaDrive/jocelyn/Evals_Progress/`), so the reference row is free.
- **Interim substitutes that exist today**: `short_noisy/treble10_speech.yaml` (noisy),
  `short_en/openasr_ami.yaml` (meeting / far-field),
  `short_en/openasr_librispeech_test_other.yaml` (harder read speech).
- **`crs_sc`** is already the strongest OOD probe in the grid: unsupported by both Whisper and
  TinyAya, 1,602 h available, six models run, and its train/val/test splits all carry the
  duration-consistency cleaning. Keep it framed as the encoder-and-decoder-unseen quadrant
  rather than as an ordinary language.
- **Long-form** is a further axis if wanted: `configs/datasets/long_ml/` has 96
  `espnet/floras` configs including Tamil, with `long_form: true`.

## 5. Lower-priority backlog — recorded, not scheduled

- **Belebele-FLEURS MCQ** (`configs/datasets/slu_en/belebele_fleurs_eng.yaml`, only `eng_Latn`
  wired; the loader is language-generic). Two blockers if revisited:
  `qasr/eval/slu.py` supports `{sentiment, qa_mcq, qa_generative}` and **text-only /
  audio-only controls do not exist** — every prompt builder hardcodes the audio block — and
  there is no scored SLU runner, only `scripts/tests/test_slu.sh` at `--max_eval_samples 4`.
  `belebele_fleurs_utils.py` does expose a `strategy ∈ {best, worst, random}` audio-selection
  knob, which is a ready-made difficulty dial.
- **MELD sentiment** — config exists (`slu_en/meld_sentiment.yaml`).
- **llama-questions** — config exists but has no `task:` field, so it currently runs as plain
  ASR/WER, not QA. Also proposed as an in-training prompt-collapse diagnostic.
- **Speech-MASSIVE** — no config, no mention anywhere in the framework.
- **IndicVoices / Kathbath** — no configs; access was gated, later granted, still unwired.
- **WaxalNLP** — no configs; `dag_asr` (Dagbani) has a known upstream `CastError`.

## Recommended order

1. Duration-consistency check on the `ta_in`/`ta_lk` configs (blocks trusting the largest
   region-match term).
2. Resolve the WorldSpeech Opus/libsndfile decode error, then add the eight `test` configs.
3. Re-run the existing grid's evaluation on both domains, in-domain and FLEURS.
4. Zero-shot cross-language sweep on FLEURS.
5. Add EdAcc.

Tiers 3 and 4 are evaluation-only and need no retraining, so they can run against the existing
checkpoints as soon as tier 1 lands.
