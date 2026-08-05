# Findings: what this study supports, and what to write

Merges the results assessment (2026-07-30) and the paper assessment (2026-08-02). Every number
is derived from a CSV in `results_all/acc/` and re-checked by `verify_paper_numbers.py`;
nothing here is hand-computed. External literature is sourced in `RELATED_WORK.md`; what to run
next is in `ROADMAP.md`.

---

# Part 1 — What to write

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
8. **`ha_ng`'s in-domain number is not held out.** Its checkpoint was selected on
   `disco-eth/WorldSpeech ha_ng test`, the same split the eval sweeps report as in-domain.
   Every other cell selected on a different split (FLEURS validation vs test; ERISLab
   val_clean vs test_clean). `ha_td` is the unbiased substitute.

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
what makes `docs/ROADMAP.md` (part 2) a real gap rather than a nice-to-have. And their
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


---

# Part 2 — The evidence

## 1. Executed vs planned

| | Planned | Actually run |
|---|---|---|
| Encoder | Whisper-small | Whisper-**medium** |
| Decoders | 5 TinyAya variants | **4** grid-wide (earth/fire/global/water); `base` and `Qwen3-4B` only on `crs_sc` |
| Languages | 6 — Hindi, Malayalam, Amharic, Shona, English, French | **10** — `crs_sc, en_us, es_419, fr_fr, ha_ng, hi_in, id_id, mr_in, sw_ke, ta_in`. Malayalam, Amharic and Shona were never run |
| Primary task | Belebele-FLEURS MCQ accuracy | **ASR / CER only** — zero MCQ runs |
| Training data | WaxalNLP / IndicVoices / FLEURS | `disco-eth/WorldSpeech` (+ `ERISLab/WorldSpeech` for `crs_sc`) |
| Recipe | connector-only SLAM | matches — `freeze_encoder=freeze_decoder=True`, `peft=False`, 1000 steps, effective batch 512, lr 1.5e-3 cosine |

The language set changed almost completely, and the primary metric was never collected.
Non-ASR evaluation is now explicitly deprioritised, so the rest of this assessment does not
argue for restoring the SLU framing; it argues for making the ASR-side claim as strong as
ASR data permits.

## 2. The plan's own gate, and where the executed grid sits against it

The publication assessment states the gate three times, in three forms: *"A leaderboard
alone caps you at a workshop"*; *"A alone is a workshop; A+one of B/C/D is a credible short
paper"*; and Phase 5's *"ship at least one, ideally two — this is what lifts the venue"*.

What ran is a leaderboard, and a partial one: one seed, four of five decoders, no
non-regional anchor outside `crs_sc`, and none of the three languages the plan named. On the
plan's own criterion this is below the workshop bar until an analysis layer is added.

The good news is that the analysis layer the plan wanted is largely *already latent in the
logs*, and section 5 is about extracting it rather than buying it with GPU hours.

## 3. The design win nobody planned for

The 10 executed languages span all three TinyAya regions:

- **earth** (African): `ha_ng`, `sw_ke`
- **fire** (South Asian): `hi_in`, `mr_in`, `ta_in`
- **water** (APAC / West Asia / Europe): `en_us`, `es_419`, `fr_fr`, `id_id`

Because all four decoders were run on every language, the **region-mismatched arm exists for
free**. The publication assessment called this arm *"critical — it converts 'specialized is
better' into 'specialization must match the language,' which is the falsifiable, interesting
claim."* The original plan proposed to add it deliberately; the executed grid already has it.

This should be the spine of the analysis. It is the only part of the executed work that is
structurally novel rather than a benchmark table.

## 4. The honest finding: region matching is a null, and more data made that clearer

`results_all/acc/t2_region_match_stats.csv`, metric = best CER (negative favours matched).
Nothing is excluded, so the primary and sensitivity analyses coincide:

| contrast | n | mean Δ | 95 % CI | Wilcoxon p | min. detectable effect |
|---|---|---|---|---|---|
| matched vs mismatched | 11 | -0.61 | [-3.95, 2.53] | 1.000 | 4.89 |
| matched vs global | 11 | -0.72 | [-3.86, 2.21] | 0.638 | 5.77 |

A Wilcoxon p of exactly 1.000 on the primary contrast is about as flat as a null gets. The
minimum detectable effect is 4.89-5.77 CER, so the design still cannot resolve a small effect --
but the point estimate is now small *and* stable across three rounds of added data.

### Nothing is excluded any more

`en_us` / water was excluded as an optimisation failure until it was re-run on 2026-07-31 and
**replicated**: best 12.05 CER against 17.06 the first time, both far worse than earth, fire and
global on the identical eval set. Two independent runs that bad is an effect, not a failure, and
since water is English's *matched* variant, excluding it had been removing the grid's strongest
against-hypothesis point. The superseded first run lives under serial 1
(`rename_wandb_serial.py`), so serial 0 stays one row per cell.

### Cells that are not clean, flagged rather than pooled

- **`fr_fr`** trains Canadian French, evaluates European French -- the one different-accent cell.
  `es_419` was listed here until the Spain-Spanish runs were deleted and re-run from `es_mx`,
  which is inside the Latin American variety FLEURS `es_419` evaluates. No longer a mismatch.
- **Interleaving is NOT a confound.** The loader uses `all_exhausted_without_replacement`, so a
  combined stream is exactly the sum of its parts. Do not re-raise it.

### 4.1 The 30 s cap, and the volume hypothesis that did not survive

**The bug.** `make_audio_length_filter_fn` keeps a clip when `length < max_input_length` -- a
*strict* comparison -- against `max_input_length: 30`. WorldSpeech `ta_lk` is pre-segmented into
fixed 30-second windows (100/100 sampled rows at exactly 30.00 s), so every clip fails
`30.0 < 30`. Filtering the interleaved Tamil stream leaves exactly **8,846** rows -- `len(ta_in)`
-- against an intended **32107**: 23,261 clips, 72.4 % of the intended Tamil training data,
silently discarded. Only `ta_lk` is totally affected; `fr_ca` loses ~4 %; `am_et` and `ur_pk`
sample 0/100 at the cap. Written up in `docs/UPSTREAM_FIXES.md`. The epoch-based reconstruction
inferred **8825** samples against a true 8,846 -- 99.76 % -- which is why that accounting is
trusted elsewhere.

Tamil is **not** excluded: the contrast is within-language, so the loss reduced every arm equally.

**The hypothesis, and its death by replication.** With seven languages the matched-decoder
benefit decayed monotonically with training-stream size, and it looked strong. Adding languages
to fill the middle and low tiers was meant to test that. It did:

| languages | Spearman rho | p |
|---|---|---|
| 7 | 0.964 | 0.0005 |
| 10 (+ am_et, en_us re-run) | 0.721 | 0.0186 |
| **11 (+ ur_pk)** | **0.555** | **0.0767** |

Every addition shrank it, and it is now **not significant even with every language**. Without
the extreme point it falls further, to rho **0.406**, p **0.2443**. The partial correlation
controlling for baseline CER is r = **-0.100**, p **0.77** -- but with 8 residual degrees of
freedom a null there is inconclusive, not evidence of no effect. Data volume and baseline
difficulty are strongly collinear across these languages (Spearman rho **-0.818**, p
**0.0021**), which is why the cross-language comparison cannot fully separate them.

The per-language table shows why:

| language | region | tier | stream | epochs | Δ vs mismatched | baseline CER | relative Δ (%) |
|---|---|---|---|---|---|---|---|
| `ta_in` | fire | very_low | 8846 | 58.01 | -14.70 | 58.33 | -25.20 |
| `am_et` | earth | very_low | 8873 | 58.01 | 0.64 | 29.22 | 2.19 |
| `ha_ng` | earth | mid | 27255 | 18.05 | -1.19 | 34.62 | -3.42 |
| `ur_pk` | fire | low | 31079 | 16.04 | 1.44 | 20.81 | 6.90 |
| `mr_in` | fire | mid | 58201 | 8.10 | -0.44 | 14.01 | -3.14 |
| `id_id` | water | high | 101112 | 5.02 | -0.40 | 6.14 | -6.44 |
| `fr_fr` | water | high | 199151 | 2.24 | -0.43 | 6.79 | -6.33 |
| `es_419` | water | high | 205972 | 2.20 | 0.07 | 3.35 | 1.94 |
| `sw_ke` | earth | high | 302088 | 1.41 | -0.15 | 13.68 | -1.06 |
| `hi_in` | fire | high | 577382 | 1.00 | 1.06 | 12.05 | 8.80 |
| `en_us` | water | high | 666718 | 1.00 | 7.39 | 4.67 | 158.31 |

The three lowest-resource cells are `ta_in` (-14.70), `am_et` (+0.64) and `ur_pk` (+1.44).
**Two of the three favour the MISMATCHED decoder**, and Tamil -- the one that does not --
also has the grid's worst accuracy at a best CER of **34.74**. `am_et` has 8,873 training clips against
`ta_in`'s 8,846 -- essentially a controlled comparison -- with the opposite sign. So Tamil is
simply an outlier, and the apparent volume interaction was one language plus a seven-point
sample. That is a finding worth stating plainly: it is the shape a spurious small-n result has,
and it was killed by exactly the data that was collected to test it. The largest single
give-back after the optimum is **46.05** CER; the OOD `crs_sc` cell's best variant reaches
**17.39** CER.

### 4.2 What the loss curves add -- and here data volume DOES predict something

`t6_loss_by_axis.csv` separates two things CER cannot. The generalisation gap
(`eval_loss_final - train_loss_final`) is inflated by overfitting *and* by distribution shift;
the eval-loss rise after its own minimum is overfitting alone.

- **Overfitting is monotone across all four resource tiers**, now that Urdu fills the low tier.
  Median eval-loss rise: very_low 0.195, low 0.188, mid 0.028, high 0.003. Time to the best eval
  loss, as a fraction of the run: 0.275, 0.289, 0.629, 0.815. Low-resource cells peak a quarter
  of the way in and then get worse for the remaining three quarters.
- **Domain shift is separable from it.** Cross-domain evals show a 6x larger generalisation gap
  than in-domain (0.196 vs 0.032) with a far smaller eval-loss rise (0.009 vs 0.000) -- the gap
  is measuring domain transfer, not overfitting.

The contrast between 4.1 and 4.2 is the real result: **training-data volume strongly predicts how
badly a cell overfits, and does not predict whether a region-matched decoder helps it.**

## 5. What the existing logs already support, at zero GPU cost

### 5.1 Sample efficiency — the Phase-5 scaling sweep, already run

The plan scoped an audio-hour scaling sweep (Phase 5, analysis D) as *new* runs at 1/5/10/25/50
hours. It is unnecessary: evaluating every 10 steps already traces each run's full curve from
~21 to ~2,700 hours of processed audio, at 101 points per run.

Ranking variants by hours-to-reach-1.5×-best-CER (`t1_sample_efficiency.csv`, mean rank over
8 languages, lower is faster):

| variant | mean rank |
|---|---|
| fire | 2.08 |
| water | 1.88 |
| earth | 2.96 |
| global | 3.08 |

This separates the variants far more cleanly than final CER does, because it uses 101 points
per run instead of 1. Note also that the ordering is **not** the same as the accuracy ordering
(by best CER: earth 2.08, fire 2.50, water 2.67, global 2.75) — *how fast* a connector learns
and *how well* it ends up are behaving as separate axes here. That is a more interesting
observation than either ranking alone, and it is free.

Both rankings put `global` last, which is the one result pointing towards regional
specialisation mattering at all. It is a rank statistic over 8 languages with one seed, so it
should be presented as suggestive, not established.

### 5.2 Overfitting — the best-vs-final gap

The last checkpoint is frequently not the best one. Mean final-minus-best CER is **3.52**. Any table quoting last-step CER is reporting a partly
arbitrary point on a curve.

The gap is worth reporting as a metric in its own right: it measures how much a run gives back
after its optimum, and it is largest exactly where a language cannot fill a clean epoch. That
is a directly testable statement once exact hours exist (section 7).

### 5.3 The OOD probe — `crs_sc`

Seychellois Creole is officially supported by **neither** Whisper nor TinyAya, making it the
only cell where both components are outside their coverage. It is also the only language with
six models, including the non-Aya `Qwen3-4B` control.

Across the five finished TinyAya variants, best CER spans **19.61 to 21.49** — a spread of
**1.88**, against per-run late-training noise of 0.62–1.65. In other words, when neither the
encoder nor the decoder has seen the language, **which regional variant is chosen barely
matters**. That is a clean, interpretable negative result and a genuine contribution: it bounds
where decoder specialisation can help.

The non-Aya control changes the reading, and it is now finished: **Qwen3-4B reaches 17.39 CER,
better than every TinyAya variant**, which widens the all-six-model spread to **4.10**. So on a
language neither component has seen, the choice that matters is the decoder *family*, not its
regional variant — the same shape as the between-family effect in `RELATED_WORK.md`, and a
second data point for the study's central claim that the treatment here is too small to measure.
State which aggregation is meant: 1.88 across the five TinyAya variants, 4.10 across all six.

Recorded on the table itself: these runs train on `train_val_exc_clean` and evaluate on
`val_clean`, and both carry the same duration-consistency cleaning as `test_clean` — samples
whose decoded audio length disagrees with the corpus `duration` column by 1 s or more are
removed. The committed example script only demonstrates the filter on the test split, so the
train/val cleaning is not visible there, but it was applied when the splits were built.

## 6. What is missing before any causal wording is defensible

1. **The matched-variant premise is still unverified** (the plan's own risk 1, still open in
   every document). The four regional variants report an identical parameter count
   (3,656,222,720) while `base` differs by exactly 16,384 — consistent with matched variants
   plus an embedding difference in `base`, but it must be confirmed against the Tiny Aya
   report. It is the cheapest item on the list and it gates the framing.
2. **A domain confound sits underneath every number.** Eight of ten languages train on
   WorldSpeech (parliamentary/broadcast) and evaluate on FLEURS (read speech). The grid
   therefore conflates decoder specialisation with domain transfer. See
   `ROADMAP.md` (part 1).
3. **No seeds.** Every claim rests on a single run per cell.
4. **No non-regional anchor outside `crs_sc`.** `base` was run for one language only, so the
   cleanest specialisation contrast (`base` vs `global`) is unavailable grid-wide.

## 7. Recommendations, in descending value per GPU-hour

1. **Treat Tamil as a small-data regime, not a defect.** Its data is verified clean (§4.1),
   so nothing needs fixing there. What remains is that Tamil has the least training data and
   the largest region-match term, which makes that term regime-specific rather than evidence
   about decoder specialisation. The cheap response is to report it that way, not to
   investigate further. `verify_dataset_durations.py` now exists so that any future suspicion
   of this kind is tested against the real data before it is written down.
2. **Reframe the headline on sample efficiency** (5.1), not endpoint CER. Zero new compute.
3. **Report the best-vs-final gap as an overfitting metric** (5.2). Zero new compute.
4. **Keep the two error-bar routes distinct.** Late-training sd measures optimisation noise and
   is available now. A bootstrap over the 210–625 eval utterances measures sampling noise and
   needs per-utterance outputs, which wandb does **not** store — they must come from the
   results directories or a re-run. They answer different questions; conflating them is a defect.
5. **Add matched-domain evaluation** so decoder effects are not read off a domain shift (§6.2).
6. **Three seeds on two or three languages** (~6–9 runs) — the cheapest new compute that
   converts "we cannot detect an effect" into "the effect is below X CER with 95 % confidence".
7. **Finish the grid**: `es_419` (`water` was never launched), relaunch `en_us`/`water`, and add
   `base` beyond `crs_sc`.
8. **Report the multi-config construction** for `ta`, `ha`, `sw` — not as a sampling defect
   (it is not one) but because the published per-config hour counts become lower bounds on the
   combined stream, which changes what a corpus-size comparison can claim.
9. **Tokenizer fertility** — still the cheapest mechanism, pure CPU, and it converts a number
   into an explanation. With exact hours available, also correlate gains against unique hours
   and repetition, which are the likeliest confounds in this grid.
10. **Lower priority, recorded so it is not rediscovered**: Belebele-FLEURS MCQ and its
    text-only/audio-only controls. `qasr/eval/slu.py` supports `{sentiment, qa_mcq,
    qa_generative}`; the controls do not exist (every prompt builder hardcodes the audio block)
    and there is no scored SLU runner, only a 4-sample smoke test.
