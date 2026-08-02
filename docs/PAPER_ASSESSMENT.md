# Honest assessment: what this study currently supports, and what to write

Written 2026-08-02, after the 12-language grid (serial 0) completed and the baseline sweep
(serial 10) began. Every number here is re-derived by `verify_paper_numbers.py`.

## Recommendation, first

**Write the 4-page workshop paper.** Not the 2-page negative-results abstract, and not the
8-page.

- **2 pages is too small**, because the study is not only a negative result. Two findings are
  positive and would have to be cut: the monotone overfitting-vs-resource-tier result, and the
  measurement of how small the "specialisation" treatment actually is.
- **8 pages is not yet supported.** It needs three things the data does not have: seeds, a
  completed baseline contrast, and in-domain evaluation. Writing 8 pages now means padding a
  single-seed grid, which is the failure mode reviewers punish hardest.
- **4 pages fits what exists**: one well-quantified null, one clean positive result, one
  mechanistic explanation for the null, and one methods contribution.

## What the study actually found

### 1. The headline hypothesis is null, and cleanly so

Region-matched decoders do not beat mismatched ones: n = 11 languages, mean **-0.61 CER**,
Wilcoxon **p = 1.000**, 95 % CI [-3.95, 2.53]. A p of exactly 1.000 is about as flat as a null
gets. The design's minimum detectable effect is **4.89 CER**.

**The null is about magnitude, not about specialisation being inert.** `arXiv:2508.05149` uses
almost this architecture and finds decoder choice matters enormously at low resource -- EuroLLM
14.0 vs Salamandra 33.6 WER at 10 h. Those are different model *families*; our variants differ by
a median of 1.30 percentage points of post-training data (§3). A null at 1.3 pp is entirely
consistent with a 19.6 pp effect between families, so the claim is "specialisation at this
magnitude does not measurably change ASR, and here is the magnitude" -- considerably stronger
than "specialisation does not help". See `docs/RELATED_WORK.md`.

**Caution on the noise floor.** Earlier drafts cited the median within-run late-training standard
deviation (1.06 CER) as the noise floor. That is the wrong quantity: it measures wobble along one
trajectory, not how far two independent runs of the same cell land apart. The single replicate
pair we have (`en_us`/water, same seed) differs by **5.01 CER** on best -- five times that figure.
`analyze_replicates.py` replaces the proxy with a measured between-run standard deviation as the
`am_et` and `crs_sc` re-runs land. Until then, every uncertainty statement here is optimistic.

### 2. A secondary hypothesis formed and then died on replication — worth reporting as such

With 7 languages, the matched-decoder benefit decayed monotonically with training-data volume
(Spearman rho 0.964, p 0.0005, robust to dropping the extreme point). Adding languages to test
it killed it:

| languages | rho | p |
|---|---|---|
| 7 | 0.964 | 0.0005 |
| 10 | 0.721 | 0.0186 |
| 11 | **0.555** | **0.0767** |

The three lowest-resource cells are `ta_in` (-14.70), `am_et` (+0.64) and `ur_pk` (+1.44), so
**two of three favour the mismatched decoder**. `am_et` at 8,873 training clips against
`ta_in`'s 8,846 is nearly a controlled comparison with the opposite sign. Tamil is an outlier;
the trend was one language plus a small sample.

### 3. The mechanistic explanation for the null: the treatment is ~1 percentage point

This is the most useful thing the study has, and it comes from the Tiny Aya report's own
appendix rather than from our runs. The matched variant saw a median of **1.30 percentage
points** more of the target language than the mismatched variants did. For the three African
languages it is 1.3 % against global's 1.0 % — a 0.3 pp difference.

So the null is a **weak-treatment null**, not a demonstration that decoder specialisation
cannot matter. That reframing is what makes the negative result publishable rather than
uninformative, and it is a claim about the Tiny Aya family that anyone building on those models
should know.

The one cell where exposure and label disagree behaves as the exposure account predicts:
English's matched variant (`water`, 17.0 % English) saw *less* English than the mismatched mean
(32.2 %, because the `fire` mix is 46.2 % English), and English is the cell where the matched
decoder is worst (+7.39). Exposure is directionally predictive overall but not significant
(Pearson r = -0.535, p = 0.09; Spearman rho = -0.239, p = 0.48).

### 4. The clean positive result: data volume predicts overfitting, monotonically

Across all four resource tiers, median eval-loss rise after its own minimum: very_low **0.195**,
low **0.188**, mid **0.028**, high **0.003**. Fraction of the run before the best eval loss:
0.275, 0.289, 0.629, 0.815. Low-resource cells peak a quarter of the way in and degrade for the
remaining three quarters.

And it is separable from distribution shift: cross-domain evaluations show a ~6x larger
generalisation gap (0.196 vs 0.032) with a near-zero eval-loss rise, so the gap measures domain
transfer while the rise measures overfitting. Reporting both is what makes that separation
visible.

**The contrast is the paper's spine:** training-data volume strongly predicts *how badly a cell
overfits*, and does not predict *which decoder to pair with it*.

### 5. A methods contribution that is genuinely reusable

A strict `<` against `max_input_length: 30`, applied to a corpus pre-segmented into fixed
30-second windows, silently discarded **72.4 %** of the Tamil training data (23,261 of 32,107
clips) and produced a -14.70 CER pseudo-effect that was on course to be the study's headline.
Any work using fixed-window corpora with a duration cap is exposed. The screen that catches it
is in the repo and runs offline.

## Strengths

- **A null with a stated minimum detectable effect**, not an unbounded "no significant
  difference".
- **The treatment size is measured**, so the null is interpretable rather than merely reported.
- **A hypothesis was formed, tested with new data, and abandoned** — visible across three
  rounds. That is unusual to see written down and it is defensible.
- **Reproducibility is real**: one command rebuilds every table and figure, with 111 unit
  checks, 24 ordering claims and 94 numbers re-derived from their CSVs. Several errors in this
  analysis were caught by that harness rather than by review.
- **Within-language contrasts throughout**, so per-language differences in eval set, difficulty
  and tokenisation cannot bias the primary comparison.
- **An OOD probe** (`crs_sc`, unseen by both encoder and decoder) and a **non-Aya control**
  (Qwen3-4B), both already run.

## Weaknesses, in the order a reviewer will raise them

1. **One seed.** MDE 4.89 CER against effects plausibly under 1 CER. This is the single biggest
   limitation and it caps what any framing can claim.
2. **The primary hypothesis is null and the secondary died.** The paper's positive content is
   the overfitting result and the treatment-size measurement, which is thinner than the
   original plan promised.
3. **A domain confound runs through everything.** 10 of 12 languages train on WorldSpeech and
   evaluate on FLEURS, so most numbers are domain-transfer numbers. The in-domain configs are
   written but not yet run.
4. **ASR only.** No SLU or MCQ, so the study cannot separate "what the model can hear" from
   "what it can say" — which was the original research question.
5. **Baselines incomplete.** Serial 10 is 2 runs in; without it there is no answer to whether
   connector training beats a downloadable model at all.
6. **Single recipe, single encoder, single decoder scale.** No claim generalises beyond
   whisper-medium + Tiny Aya 3.35B + connector-only.
7. **`am_et` and `crs_sc` have no uploaded checkpoints**, so they cannot enter the eval sweep.

## Where this sits in the literature

`arXiv:2508.05149` ("Speech LLMs in Low-Resource Scenarios") is a near-twin -- frozen
Whisper-large-v3-turbo, frozen LLM, trained linear projector -- and supplies two anchors that were
not available when this assessment was first written:

- **The 100-200 h threshold.** SLAM-ASR only matches a Whisper-only baseline at 100-200 h of
  training data (10 h -> 14.0 WER, 100 h -> 7.6, 200 h -> 6.4, against Whisper 7.1). **Five of our
  twelve cells sit below it.** This explains our low-resource cells better than anything we derived
  ourselves, and it belongs ahead of any claim about decoder variants at low resource.
- **Independent corroboration of the domain problem.** At 200 h their in-domain WER is 6.4 but
  FLEURS is 13.2, against Whisper's 5.8. We see the same from the other side: WorldSpeech test is
  2.40x (en_us) and 1.89x (fr_fr) harder than FLEURS test for whisper-medium.

They also use **no augmentation at all** despite naming data scarcity as the bottleneck, which is
what makes `docs/FUTURE_WORK_AUGMENTATION.md` a real gap rather than a nice-to-have. And their
cross-lingual projector pretraining result (14.0 -> 8.6 WER in-domain at 10 h) is the strongest
known next lever for our low tiers.

## Proposed 4-page structure

1. **Intro** — the question, and why Tiny Aya's matched variants make it answerable.
2. **Setup** — 12 languages x 4 variants, connector-only, the resource-tier ladder.
3. **Result 1, the null** — with MDE, and immediately the treatment size (~1.3 pp), which is
   what makes the null mean something.
4. **Result 2, the positive** — overfitting monotone in resource tier, separated from domain
   shift by the two loss quantities.
5. **What we tried and abandoned** — the volume-interaction hypothesis and its death across
   7 -> 10 -> 11 languages. Short, and it buys credibility.
6. **Methods note** — the duration-cap data loss, with the screen.
7. **Limitations** — seeds first, honestly.

## What would justify 8 pages

In descending order of value per GPU-hour:

1. **3 seeds on 3 languages** (~12 runs). Drops MDE from ~4.9 toward ~1 CER. Without this no
   framing can make a strong claim, and with it the null becomes a genuinely tight bound.
2. **Finish serial 10 and run serial 11** — the like-for-like "what did connector training
   buy" contrast on identical FLEURS test configs. This is the result most readers will want
   first, and both scripts are written.
3. **Run the in-domain WorldSpeech evals**, removing the domain confound from every number.
4. **A stronger treatment.** The natural follow-up given finding 3: if specialisation is ~1 pp
   of post-training data, then the interesting experiment is a decoder with substantially more
   target-language data, not another regional variant.

Items 2 and 3 need no new training, only evaluation.
