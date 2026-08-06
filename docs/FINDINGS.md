# Findings: the evidence behind the paper

The paper is `ACL26_LALMDecoder/main.tex`; this is the working record its Results
section is drawn from. Framing advice that predated the paper has been removed.

Merges the results assessment (2026-07-30) and the paper assessment (2026-08-02). Every number
is derived from a CSV in `results_all/acc/` and re-checked by `verify_paper_numbers.py`;
nothing here is hand-computed. External literature is sourced in the paper's related work; what to run
next is in `ROADMAP.md`.

---

# Part 1 — What to write

## What the study actually found

### 1. The headline hypothesis is null, and cleanly so

Region-matched decoders do not beat mismatched ones: n = 11 languages, mean **-0.55 CER**,
Wilcoxon **p = 0.966**, 95 % CI [-3.89, 1.99]. The design's minimum detectable effect is
**4.90 CER**, so the point estimate sits well inside what the design cannot resolve.

**The null is about magnitude, not about specialisation being inert.** `arXiv:2508.05149` uses
almost this architecture and finds decoder choice matters enormously at low resource -- EuroLLM
14.0 vs Salamandra 33.6 WER at 10 h. Those are different model *families*; our variants differ by
a median of 1.30 percentage points of post-training data (§3). A null at 1.3 pp is entirely
consistent with a 19.6 pp effect between families, so the claim is "specialisation at this
magnitude does not measurably change ASR, and here is the magnitude" -- considerably stronger
than "specialisation does not help". See the paper's related work.

**The noise floor is now measured, and it is close to the proxy.** With the `am_et` and `crs_sc`
re-runs landed there are **10 replicate pairs**, giving a between-run standard deviation of
**1.17 CER** against a median within-run `late_sd` of **1.09**. An earlier draft, working from the
single `en_us`/water pair that differs by 5.01 CER, concluded the proxy was optimistic by ~5x.
Nine further pairs do not support that: they average **1.51 CER** apart with a maximum of 3.21, so
`en_us`/water is the outlier rather than the rule. `late_sd` turns out to be a defensible proxy,
and the minimum detectable effect stands roughly where it was.

Nine of the ten pairs vary the seed (42 against 420) and so estimate the larger quantity --
seed sensitivity, not just nondeterminism. Their between-run sd is **0.90 CER**.

### 2. A secondary hypothesis formed and then died on replication — worth reporting as such

With 7 languages, the matched-decoder benefit decayed monotonically with training-data volume
(Spearman rho 0.964, p 0.0005, robust to dropping the extreme point). Adding languages to test
it killed it:

| languages | rho | p |
|---|---|---|
| 7 | 0.964 | 0.0005 |
| 10 | 0.721 | 0.0186 |
| 11 | 0.555 | 0.0767 |
| 11, `am_et`/`crs_sc` re-run | **0.482** | **0.1334** |

The three lowest-resource cells are `ta_in` (-14.70), `am_et` (+1.32) and `ur_pk` (+1.44), so
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

Across all four resource tiers, median eval-loss rise after its own minimum: very_low **0.211**,
low **0.188**, mid **0.041**, high **0.003**. Fraction of the run before the best eval loss:
0.282, 0.292, 0.627, 0.816. Low-resource cells peak a quarter of the way in and degrade for the
remaining three quarters.

Both are **medians of per-language medians** over the 48-run grid, so a tier with seven languages
cannot outvote one with two. Under a per-run aggregation the sequence is not monotone, and the
per-run version was also contaminated: until the control arms moved to serial 2, `ta_in`'s `base`
and `qwen3-4b` sat in the `very_low` tier.

**A claim that did not survive the clean grid.** Earlier drafts reported a ~6x larger
generalisation gap cross-domain (0.196 vs 0.032) and read it as the gap measuring domain
transfer. On the 48-run grid the two domains are indistinguishable on that measure: **0.184
cross-domain against 0.179 in-domain** per-language, and 0.190 against 0.173 per-run. The 6x was
an artifact of the control arms, whose six near-zero `crs_sc` runs outnumbered `ha_ng`'s four in
the in-domain pool. Only two languages evaluate in-domain, so this comparison is underpowered in
either direction and should not be presented as a result.

What does still separate is the eval-loss rise: **0.009 cross-domain against 0.001 in-domain**,
both near zero. That supports reporting the rise as the overfitting measure, and it is why the
tier result above rests on the rise rather than on the gap.

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

1. **One seed per cell across most of the grid.** MDE 4.90 CER against effects plausibly under
   1 CER. Ten cells now have a second run, which is what made the noise floor measurable, but the
   other 42 do not. This is the single biggest limitation and it caps what any framing can claim.
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
   val_clean vs test_clean). The analysis uses `ha_td` as Hausa's in-domain point instead;
   `ha_ng` is reported as accent transfer.

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
| matched vs mismatched | 11 | -0.55 | [-3.89, 1.99] | 0.966 | 4.90 |
| matched vs global | 11 | -0.50 | [-4.53, 2.42] | 0.966 | 5.83 |

A Wilcoxon p of 0.966 on the primary contrast is about as flat as a null gets. The minimum
detectable effect is 4.90-5.83 CER, so the design still cannot resolve a small effect -- but the
point estimate is small *and* stable across four rounds of added data.

### Nothing is excluded any more

`en_us` / water was excluded as an optimisation failure until it was re-run on 2026-07-31 and
**replicated**: best 12.05 CER against 17.06 the first time -- the widest of the ten replicate
pairs, whose best-CER values reach 30.38 on the first run and 31.01 on the re-run. Both are far
worse than earth, fire and
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

### 4.1 The volume hypothesis that did not survive

Tamil trains on `ta_in` alone: **8,846** clips, the smallest stream in the grid. The epoch-based
reconstruction infers **8825** against that true 8,846 -- 99.76 % -- which is why the accounting
is trusted elsewhere.

Tamil is **not** excluded: the contrast is within-language, so the loss reduced every arm equally.

**The hypothesis, and its death by replication.** With seven languages the matched-decoder
benefit decayed monotonically with training-stream size, and it looked strong. Adding languages
to fill the middle and low tiers was meant to test that. It did:

| languages | Spearman rho | p |
|---|---|---|
| 7 | 0.964 | 0.0005 |
| 10 (+ am_et, en_us re-run) | 0.721 | 0.0186 |
| 11 (+ ur_pk) | 0.555 | 0.0767 |
| **11, am_et/crs_sc re-run** | **0.482** | **0.1334** |

Every addition shrank it, and it is now **not significant even with every language**. Without
the extreme point it falls further, to rho **0.309**, p **0.3848**. The partial correlation
controlling for baseline CER is r = **-0.129**, p **0.71** -- but with 8 residual degrees of
freedom a null there is inconclusive, not evidence of no effect. Data volume and baseline
difficulty are strongly collinear across these languages (Spearman rho **-0.818**, p
**0.0021**), which is why the cross-language comparison cannot fully separate them.

The per-language table shows why:

| language | region | tier | stream | epochs | Δ vs mismatched | baseline CER | relative Δ (%) |
|---|---|---|---|---|---|---|---|
| `ta_in` | fire | very_low | 8846 | 58.01 | -14.70 | 58.33 | -25.20 |
| `am_et` | earth | very_low | 8873 | 58.01 | 1.32 | 27.93 | 4.73 |
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
  Median eval-loss rise: very_low 0.211, low 0.188, mid 0.041, high 0.003. Time to the best eval
  loss, as a fraction of the run: 0.282, 0.292, 0.627, 0.816. Low-resource cells peak a quarter
  of the way in and then get worse for the remaining three quarters.
- **The generalisation gap does NOT separate the domains** on the clean grid: 0.184 cross-domain
  against 0.179 in-domain. The eval-loss rise does, weakly (0.009 vs 0.001), and only two
  languages evaluate in-domain, so neither comparison carries much weight.

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
| fire | 2.00 |
| water | 2.08 |
| earth | 2.75 |
| global | 3.17 |

This separates the variants far more cleanly than final CER does, because it uses 101 points
per run instead of 1. Note also that the ordering is **not** the same as the accuracy ordering
(by best CER: earth 2.17, fire 2.50, water 2.67, global 2.67) — *how fast* a connector learns
and *how well* it ends up are behaving as separate axes here. That is a more interesting
observation than either ranking alone, and it is free.

Both rankings put `global` last, which is the one result pointing towards regional
specialisation mattering at all. It is a rank statistic over 8 languages with one seed, so it
should be presented as suggestive, not established.

### 5.2 Overfitting — the best-vs-final gap

The last checkpoint is frequently not the best one. Mean final-minus-best CER is **5.36**. Any
table quoting last-step CER is reporting a partly arbitrary point on a curve.

The gap is worth reporting as a metric in its own right: it measures how much a run gives back
after its optimum, and it is largest exactly where a language cannot fill a clean epoch. That
is a directly testable statement once exact hours exist (section 7).

### 5.3 The OOD probe — `crs_sc`

Seychellois Creole is officially supported by **neither** Whisper nor TinyAya, making it the
only cell where both components are outside their coverage. It is also the only language with
six models, including the non-Aya `Qwen3-4B` control.

Across the five finished TinyAya variants, best CER spans **17.92 to 21.35** — a spread of
**3.43**, against per-run late-training noise of 0.62–1.12. In other words, when neither the
encoder nor the decoder has seen the language, **which regional variant is chosen barely
matters**. That is a clean, interpretable negative result and a genuine contribution: it bounds
where decoder specialisation can help.

The non-Aya control changes the reading, and it is now finished: **Qwen3-4B reaches 17.39 CER,
better than every TinyAya variant**, which widens the all-six-model spread to **3.96**. So on a
language neither component has seen, the choice that matters is the decoder *family*, not its
regional variant — the same shape as the between-family effect in the paper's related work, and a
second data point for the study's central claim that the treatment here is too small to measure.
State which aggregation is meant: 3.43 across the five TinyAya variants, 3.96 across all six.

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
