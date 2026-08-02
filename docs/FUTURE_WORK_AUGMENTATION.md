# Future work: can augmentation substitute for training hours?

**Status: NOT current work.** Recorded 2026-08-03 so the analysis is not lost. Nothing here is
scheduled, and nothing here has been built. The current study works only with data already
collected.

---

## Why this is worth doing later

`arXiv:2508.05149` ("Speech LLMs in Low-Resource Scenarios") is the closest prior work to this
study — frozen Whisper-large-v3-turbo + frozen LLM + trained linear projector, Italian scaled
10 → 252 h. It establishes that **the SLAM-ASR framework needs 100–200 h** to match a Whisper-only
baseline, and it **uses no augmentation at all**, without discussing it, despite naming data
scarcity as the core bottleneck.

**Five of our twelve cells sit below that threshold**: `ta_in` (~35 h), `am_et` (~38 h), `ur_pk`
(~65 h), and the mid tier `ha_ng` / `mr_in` (~110–120 h). So "can augmentation substitute for
hours" is open, well-motivated, and unaddressed by the nearest neighbour in the literature.

A second motivation is already in this project's own record: `SLAM-ASR_TaskOverfitting_Problem`
flags SALMONN-style task overfitting as a *demonstrated* failure mode here — a connector trained
on one fixed ASR prompt stops following other instructions. Arm B below attacks that and the
data-scarcity question with the same intervention.

## A correction that motivated this document

I previously claimed text augmentation was structurally infeasible with a frozen decoder, on the
grounds that "text with no audio has no gradient path". That answered the wrong question.

Keeping the audio and changing the **target** — `(audio, "summarise this", summary)` instead of
`(audio, "transcribe", transcript)` — backpropagates into the connector perfectly well with
encoder and decoder frozen, because the forward pass still runs audio → connector → decoder and
the loss lands on the new response tokens. It is feasible, and the infrastructure already exists
upstream.

What genuinely *is* infeasible is text with **no audio at all** (LM-style adaptation), which needs
decoder LoRA — see the "not an arm" note at the end.

## The three arms

| arm | audio | target | feasible with frozen decoder? | build cost |
|---|---|---|---|---|
| **A. Audio augmentation** | perturbed (RIR / noise / speed) | unchanged transcript | yes | new module |
| **B. Target augmentation** | unchanged | synthesised tasks | yes | **already exists upstream** |
| **C. Both** | perturbed | synthesised tasks | yes | A + B |

The arms separate two different things augmentation could supply: **A** adds acoustic variation
with the same linguistic content; **B** adds linguistic and task variation over the same acoustics.
If only A helps, the bottleneck is acoustic coverage; if only B helps, it is supervision density
per audio hour. That distinction is the actual research question, and it is why running one arm
alone would be uninformative.

### Arm B already exists

`tools/preprocess/augment_lalm_sft_data.py` (QuantizedASR) uses a text-LLM teacher (Qwen3-8B) over
each transcript to synthesise 15 task types — `open_qa`, `closed_qa`, `extractive_qa`, `summarize`,
`gist`, `topic_genre`, `keywords`, `entities`, `title`, `paraphrase`, `translate`, `continue`,
`simplify`, `speaker_sex`, `sentiment` — adding `instruction` / `response` columns. **Audio and the
original transcript are untouched**, and `asr_transcribe` is kept as a deliberately low-weight
(0.5) anchor so the model does not collapse to transcription.

Wiring already present upstream:
- label side — `qasr/data/data_utils.py:184-192` swaps the target to `response` when an
  `instruction` column exists;
- prompt side — `qasr/data/collators.py:404-410` builds the instruction prompt.

Our runs log `model_type: qwen2_audio`, so `QwenAudioCollator` applies and **both** hooks work.
(`AudioFlamingoCollator` ignores `samples`, so on an AF3-style model only the label side would
function — not our case.)

Adaptation needed: it reads LibriSpeech-style columns (`text`, `speaker_id`, `id`) while
WorldSpeech uses `human_transcript`.

### Arm A does not exist at all

No SpecAugment, noise mixing, RIR convolution, speed perturbation or `torchaudio` transform exists
anywhere outside the vendored `transformers/` copy. SpecAugment is not even *implemented* in the
Qwen2-Audio encoder path, so flipping a config flag would be a no-op.

**Insertion point:** `BaseASRCollator._extract_audios` (`qasr/data/collators.py:140`) — the single
funnel every waveform passes through as 16 kHz float32, immediately before the feature extractor.
Three reasons it is the right place:

1. **Train/eval separation is already free.** Two collator objects are built, and
   `qasr/train/train_utils.py:139` selects between them, so `if not self.inference:` is a
   leak-proof gate.
2. **Randomness is fresh per batch**, so every epoch sees a different realisation.
3. **It runs in dataloader workers**, so CPU-side DSP parallelises at no cost.

**Not `prepare_data`**: it takes no `train` argument, and the local-dataset path loads eagerly
(`qasr/data/data_utils.py:261-270`), so a `.map()` transform would be cached into a single frozen
realisation — no per-epoch variation, which defeats the purpose.

## Blockers and confounds, recorded before they bite

- **`WhisperCollator` has no `inference` field**, so the train/eval gate is not leak-proof for a
  pure-Whisper run until one is added. An augmentation leaking into eval would invalidate every
  number it touched.
- **Speed perturbation changes clip duration**, and would interact with the strict `< 30 s`
  `max_input_length` filter — a 1.1× slow-down can push a 27 s clip past the cap and silently drop
  it. That is precisely the failure mode that cost 72 % of the Tamil training data.
- **Teacher-quality confound in arm B.** The teacher is a *text* LLM, and its coverage of Amharic,
  Hausa and Urdu is weaker than of English — so teacher quality co-varies with the exact axis under
  study. Needs an in-language versus English-target control, or the arm confounds itself.
- **Licensing.** `treble-technologies/Treble10-RIR` (1K–10K RIRs, 6-channel and ambisonic) is
  **CC-BY-NC-SA-4.0** — non-commercial and share-alike — unlike `Treble10-Speech`'s CC-BY-4.0.
  Derived augmented data inherits share-alike, which constrains release.

## A reference arm would be required

`arXiv:2508.05149` shows cross-lingual projector pretraining is large and proven: at 10 h Italian,
pretraining on 200 h Spanish gives **14.0 → 8.6** WER in-domain and **35.2 → 20.3** out-of-domain.

Without such an arm, a null on augmentation cannot be distinguished from "nothing would have
helped at this data budget" — which is the same trap the region-match null fell into before the
exposure magnitude was measured.

## Not an arm

True text-only adaptation, with no audio at all. `arXiv:2506.05671` implements it by fine-tuning
**LoRA inside the decoder**, and reports that done naively "the model overfits to the text modality
and loses its ability to effectively integrate acoustic representations". Adding LoRA would unfreeze
the decoder and break the "only the decoder's SFT mix differs" control that this study rests on, so
region-match results would no longer be comparable across arms.

## If it is ever scheduled

Suggested shape: the three lowest cells (`ta_in`, `am_et`, `ur_pk`) plus one high-resource control
(`id_id`), 4 variants each, arms {baseline, A, B, C, cross-lingual pretraining}. The advance
prediction worth committing to: if augmentation substitutes for hours, a ~38 h cell under arm C
should approach the 100–200 h band. If it does not, that bounds augmentation as a substitute for
data — which is itself publishable, given the gap in the closest prior work.
