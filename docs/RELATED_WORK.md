# Related work, and what it changes about our results

Written 2026-08-03. Numbers quoted from the papers are transcribed here verbatim so they can be
checked; numbers about *our* results are re-derived from CSVs by `verify_paper_numbers.py`.

---

## The near-twin: Speech LLMs in Low-Resource Scenarios

**`arXiv:2508.05149`** — "Speech LLMs in Low-Resource Scenarios: Data Volume Requirements and the
Impact of Pretraining on High-Resource Languages".

Almost the same architecture as this study: **frozen** Whisper-large-v3-turbo encoder, **frozen**
LLM decoder (EuroLLM 1.7B, and Salamandra 2B for comparison), and a **trained linear projector**
(17.31 M parameters, one hidden layer + ReLU, downsampling k=5). Italian scaled from 10 to 252 h
of Common Voice, plus a Galician case study at 10–15 h.

### 1. The data-volume threshold

> "at least 100-200 hours of labeled data are needed for the SLAM-ASR framework to match the
> performance of Whisper-only models"

| hours | CV IT WER | vs Whisper-v3-turbo (7.1) |
|---|---|---|
| 10 | 14.0 | worse |
| 15 | 13.9 | worse |
| 20 | 11.4 | worse |
| 50 | 9.2 | worse |
| 100 | 7.6 | slightly worse |
| 200 | 6.4 | **better** |
| 252 | 6.1 | **better** |

**Why this matters to us.** Five of our twelve cells sit below 200 h — `ta_in` (~35 h), `am_et`
(~38 h), `ur_pk` (~65 h), and the mid tier `ha_ng` / `mr_in` (~110–120 h). Our low tiers are
therefore squarely inside the regime where this framework is *known* to underperform a plain
Whisper baseline. That is a better explanation for those cells than anything we had derived
ourselves, and it should be stated before any claim about decoder variants at low resource.

### 2. It reframes our region-match null

Their most striking result is that **decoder choice matters enormously at low resource**, and
converges away as data grows:

| | 10 h CV IT WER |
|---|---|
| EuroLLM 1.7B | 14.0 |
| Salamandra 2B | 33.6 |

> "the performance gap between Salamandra and EuroLLM tends to close as more data are available"

That is a 19.6 pp decoder effect — against our region-match effect of **−0.61 CER** (Wilcoxon
p = 1.000). The two are not in conflict, and saying so is the single most useful thing this paper
does for our write-up:

- Their decoders are **different model families** with different pretraining corpora.
- Ours are **regional variants of one family**, differing by a median of **1.30 percentage points**
  of post-training data (`t8_exposure.csv`) — and by only 0.3 pp for the three African languages.

So our null is *consistent with* their positive result. The claim becomes **"decoder specialisation
at ~1 pp of post-training data does not measurably change ASR, and here is the magnitude"**, which
is far stronger than "specialisation does not help". It also predicts where a positive result
would live: between decoders that differ substantially, not between regional variants of one.

### 3. Their out-of-domain gap matches ours

At 200 h their in-domain WER is 6.4 but FLEURS is 13.2, against Whisper's 5.8 — they call this
"the challenge that the SLAM-ASR framework has when generalizing across domains".

We see the same shape from the other direction: our baselines show WorldSpeech test is **2.40×**
(en_us) and **1.89×** (fr_fr) harder than FLEURS test for whisper-medium, and our training cells
train on WorldSpeech while 10 of 12 evaluate on FLEURS. Independent confirmation that
cross-domain transfer, not decoder choice, is where this framework loses.

### 4. Cross-lingual projector pretraining is the proven intervention

At 10 h of Italian fine-tuning:

| pretraining source | CV IT WER | FLEURS IT WER |
|---|---|---|
| scratch | 14.0 | 35.2 |
| LS100 EN | 12.0 | 16.1 |
| CV100 EN | 9.8 | 17.1 |
| CV200 ES | 8.6 | 20.3 |

The advantage "diminishes with smaller performance gains" by 200 h (scratch 6.4 vs LS100 EN 6.6).
A multilingual projector helped Galician at 10 h: 18.6 → 13.3 CV, 43.0 → 19.4 FLEURS.

This is the strongest known lever for our low-resource cells, and it is a reference point any
future intervention should be measured against.

### 5. They use no augmentation

Confirmed absent: no SpecAugment, speed perturbation, noise, reverberation/RIR, or text
augmentation, and no discussion of augmentation as a mitigation despite naming data scarcity as
the core bottleneck. That is the gap recorded in `FUTURE_WORK_AUGMENTATION.md`.

---

## Secondary anchors

- **`arXiv:2506.05671`** — "Low-Resource Domain Adaptation for Speech LLMs via Text-Only
  Fine-Tuning". Text-only adaptation is done by fine-tuning **LoRA inside the decoder**; the
  encoder and projector are frozen and receive zero gradient. Reports 27.59 → 16.22 WER on
  SlideSpeech (473 h of target-domain text) and warns that naive text-only training makes "the
  model overfit to the text modality and lose its ability to effectively integrate acoustic
  representations". Relevant to us as the boundary of what a **frozen** decoder permits: adding
  LoRA would break the control this study rests on.
- **`arXiv:2601.20900`** (ICASSP 2026) — text-only adaptation framed as a denoising task, to
  preserve speech–text alignment while adapting to a domain.
- **`arXiv:2606.29031`** — synthetic speech for LLM-based ASR. Listed for completeness; the PDF did
  not render legibly enough to quote numbers, so nothing is claimed from it here.
- **`arXiv:1904.08779`** (SpecAugment) and Journal on Audio, Speech and Music Processing
  `s13636-026-00451-8` — establish speed perturbation + SpecAugment as the standard low-resource
  augmentation baseline, with combinations reported to beat either alone.

---

## What this repo should say as a result

1. Lead the low-resource discussion with the **100–200 h threshold**, and state that five of our
   cells are below it.
2. Present the region-match null **together with the 1.30 pp treatment size** and the contrast
   against a 19.6 pp between-family decoder effect. The null is about magnitude, not about
   specialisation being inert.
3. Keep the cross-domain finding, which now has independent corroboration.
4. Point at cross-lingual projector pretraining as the strongest known next lever, ahead of
   augmentation.
