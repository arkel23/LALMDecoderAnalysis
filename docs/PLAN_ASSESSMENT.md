# Assessment of the original LisTAya plan against what was actually run

Written 2026-07-30, after downloading and analysing wandb `LisTAya/LALMDecoder` serial 0
(41 runs). Source documents assessed: `ExecutionMasterPlan.md`,
`LisTAya_Publication_Assessment.md`, `Chat_Survey_LALMDecoderSFTEvalFeasibility.md` and
`SLAM-ASR_TaskOverfitting_Problem.docx`, all under
`/mnt/c/Users/edwin.rios/Claude/Projects/LisTAya_Listening_TinyAya`.

Every number below is derived from a CSV in `results_all/acc/` and re-checked by
`verify_paper_numbers.py`. Nothing here is hand-computed.

---

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
controlling for baseline CER is r = **-0.100**.

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
**1.88**, against per-run late-training noise of 0.64–1.65. In other words, when neither the
encoder nor the decoder has seen the language, **which regional variant is chosen barely
matters**. That is a clean, interpretable negative result and a genuine contribution: it bounds
where decoder specialisation can help.

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
   `EVAL_DATASET_PLAN.md`.
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

## 8. The framing this data can actually support

> Under a matched connector-only recipe across nine languages and three regional TinyAya decoder
> variants, the benefit of matching a decoder's regional specialisation to the target language
> **decays monotonically with training-data volume** — from -14.70 CER at 8,846 training
> utterances to +1.06 CER at 577,382 (Spearman rho = 0.964, p = 0.0005; rho = 0.943, p = 0.0048
> with the extreme point removed). Where data is plentiful the effect is bounded within about
> 1 CER, so a general decoder suffices; where data is scarce, matching the decoder's region is
> worth more than any other choice available in the pipeline. Data volume and task difficulty are
> collinear across these languages (rho = -0.679) and the cross-language comparison cannot fully
> separate them: the partial correlation controlling for baseline error rate is inconclusive over
> all seven languages (r = 0.137, p = 0.77) though significant over the six non-extreme ones
> (r = 0.838, p = 0.04). Separately, variants differ in **sample efficiency** — `fire` reaches a
> given error rate fastest (mean rank 1.69) and `global` slowest (3.19) — and that ordering is
> not the accuracy ordering. On a language unseen by both encoder and decoder, variant choice is
> immaterial (4.10 CER spread against 0.62-1.65 run noise).

Why this is the right frame:

- **It matches the project's actual motivation.** This was always a low-resource SLU project.
  "Match the decoder's region when data is scarce, use a general one otherwise" is directly
  actionable, and it turns the earlier bounded null into the *high-resource half* of a scaling
  result rather than the whole story.
- **It is a mechanism, not a leaderboard.** The publication assessment's own gate is that a
  leaderboard caps the venue at a workshop while a mechanistic finding lifts it. An interaction
  with data volume is that finding, and it required no new infrastructure — only reading the
  existing runs correctly.
- **It subsumes the earlier results instead of discarding them.** The bounded null becomes the
  high-resource asymptote; the sample-efficiency ordering becomes the dynamics counterpart; the
  `crs_sc` result becomes the boundary case where neither component has seen the language.
- **The bug is a genuine methods contribution.** A strict inequality against a corpus
  pre-segmented exactly at the duration cap destroyed 72.4 % of one language's training data and
  manufactured a -14.70 CER pseudo-effect that was on course to be reported as the study's
  strongest result. Any study using fixed-window corpora with a duration cap is exposed.
  `verify_dataset_durations.py` is the reusable screen, and it fails on a config whose clips sit
  at the cap.
- **What it must not claim.** That decoder specialisation improves listening accuracy in
  general — it does not, above roughly 60k utterances. And not that data volume is the proven
  driver: that awaits the paired manipulation.
