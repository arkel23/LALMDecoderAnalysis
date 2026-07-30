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

## 4. The honest finding: the grid cannot currently detect the effect it was built for

`results_all/acc/t2_region_match_stats.csv`, metric = best CER (lower is better; a negative
delta favours the region-matched decoder):

| analysis | contrast | n | favouring matched | mean Δ | 95% CI | Wilcoxon p | min. detectable effect |
|---|---|---|---|---|---|---|---|
| primary (failed runs excluded) | matched vs mismatched | 7 | 6 | −2.32 | [−6.55, +0.12] | 0.156 | 6.97 |
| primary | matched vs global | 7 | 6 | −2.78 | [−7.83, +0.10] | 0.156 | 8.32 |
| sensitivity (all runs) | matched vs mismatched | 8 | 6 | −0.48 | [−5.65, +4.38] | 0.461 | 8.39 |
| sensitivity | matched vs global | 8 | 6 | −0.92 | [−6.76, +4.19] | 0.461 | 9.27 |

Three readings, all of which must be stated together:

1. **The direction is consistent.** 6 of 7 (primary) and 6 of 8 (sensitivity) languages favour
   the region-matched decoder. That is suggestive and worth reporting.
2. **The interval does not exclude zero, under either analysis.** p = 0.156 and p = 0.461.
3. **The minimum detectable effect is 6.97–9.27 CER points.** With one seed and 7–8 languages,
   this design can only detect effects an order of magnitude larger than any plausible
   decoder-specialisation effect. The correct conclusion is not "there is no effect" but
   **"this grid cannot measure an effect of the size we are looking for."**

The supporting evidence for point 3 is a noise measurement, not an assumption. The median
within-run late-training CER standard deviation is **1.03**
(`t1_sample_efficiency.csv`, `late_sd`, over the four grid-wide variants) — larger than the
sensitivity-analysis mean effect of 0.48. The measurement noise exceeds the signal.

### Why the primary and sensitivity analyses differ so much

The `en_us` / `water` run is an optimisation failure: it converges to ~20 CER while
earth/fire/global reach 4.5–5.4 on the identical eval set, with a curve
(`225 → 60 → 33 → 25 → 21 → 20`) that is converged-but-bad rather than a late spike. `water`
is also the region-matched variant for `en_us`, so excluding the failed run removes that
language's matched arm entirely, taking the largest against-hypothesis point with it.

Dropping the language is defensible. Dropping it *silently* would not be — the exclusion
roughly quintuples the apparent effect (−0.48 → −2.32). Both analyses are therefore reported,
and neither is significant.

### Two cell types that are not clean, and are flagged rather than pooled

- **Cross-dialect**: `fr_fr` trains on `fr_ca` and evaluates on `fr_fr`; `es_419` trains on
  `es_es` (the running jobs' wandb config says `es_mx`, so the YAML changed after launch) and
  evaluates on `es_419`.
- **Uniform interleaving**: `ta_in`+`ta_lk`, `ha_ng`+`ha_td` and `sw_ke`+`sw_tz` are
  interleaved at `1/N` probabilities *regardless of relative corpus size*
  (`qasr/data/data_utils.py:239-251`; the size-proportional line is commented out). The
  smaller config is heavily oversampled.

`ta_in` is both interleaved and by far the largest single contributor to the region-match
effect. Its data accounting does not reconcile with the WorldSpeech report and is the first
thing that should be checked — see section 7.

## 5. What the existing logs already support, at zero GPU cost

### 5.1 Sample efficiency — the Phase-5 scaling sweep, already run

The plan scoped an audio-hour scaling sweep (Phase 5, analysis D) as *new* runs at 1/5/10/25/50
hours. It is unnecessary: evaluating every 10 steps already traces each run's full curve from
~21 to ~2,700 hours of processed audio, at 101 points per run.

Ranking variants by hours-to-reach-1.5×-best-CER (`t1_sample_efficiency.csv`, mean rank over
8 languages, lower is faster):

| variant | mean rank |
|---|---|
| fire | 1.69 |
| water | 2.19 |
| earth | 2.94 |
| global | 3.19 |

This separates the variants far more cleanly than final CER does, because it uses 101 points
per run instead of 1. Note also that the ordering is **not** the same as the accuracy ordering
(by best CER: earth 2.00, fire 2.50, water 2.50, global 3.00) — *how fast* a connector learns
and *how well* it ends up are behaving as separate axes here. That is a more interesting
observation than either ranking alone, and it is free.

Both rankings put `global` last, which is the one result pointing towards regional
specialisation mattering at all. It is a rank statistic over 8 languages with one seed, so it
should be presented as suggestive, not established.

### 5.2 Overfitting — the best-vs-final gap

The last checkpoint is frequently not the best one. Mean final-minus-best CER is **3.47**, with
a maximum of **19.59** (`ta_in` / fire). Any table quoting last-step CER is reporting a partly
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

Caveat recorded on the table itself: these runs evaluate on `ERISLab/WorldSpeech` split
`val_clean`, and the upstream cleaning pass applies its duration-consistency filter to the
`test` split only. `val_clean` is an unfiltered 0.1 % re-split of train with `audio_length_s`
materialised — the split's name promises a filter it did not receive.

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

1. **Fix the data accounting first — pure CPU, and the largest apparent effect depends on it.**
   Epoch-derived unique hours reconcile with the WorldSpeech report for only 3 of 9 languages
   (`crs_sc`, `ha_ng`, `sw_ke`) and fail badly for `ta_in` (≈35 h implied against ≥240 h
   published). `train/train_audio_seconds` itself is trustworthy — it sums decoded array length
   over sampling rate (`qasr/data/data_utils.py:71-82`) — but the wandb `train/epoch` counter is
   an estimate under `streaming=True`, so it must not be used as a divisor. Compute exact hours
   with the framework's own function, and apply the `crs_sc` duration-consistency check to
   `ta_in`. Until then, report audio *processed*, never unique hours.
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
8. **Fix or report the uniform interleaving** for `ta`, `ha`, `sw`. The size-proportional code
   already exists, commented out.
9. **Tokenizer fertility** — still the cheapest mechanism, pure CPU, and it converts a number
   into an explanation. With exact hours available, also correlate gains against unique hours
   and repetition, which are the likeliest confounds in this grid.
10. **Lower priority, recorded so it is not rediscovered**: Belebele-FLEURS MCQ and its
    text-only/audio-only controls. `qasr/eval/slu.py` supports `{sentiment, qa_mcq,
    qa_generative}`; the controls do not exist (every prompt builder hardcodes the audio block)
    and there is no scored SLU runner, only a 4-sample smoke test.

## 8. The framing this data can actually support

Not *"specialising the decoder's SFT data improves listening"* — the grid cannot resolve an
effect that size. What it can support, honestly:

> Under a fixed connector-only recipe across 10 languages and three regional decoder variants,
> region-matched decoders are directionally but not significantly better (6/7 languages,
> p = 0.16, 95 % CI [−6.55, +0.12] CER); the design's minimum detectable effect is 7 CER points,
> which bounds what a single-seed 4×10 grid can establish. Variants differ more in *sample
> efficiency* than in final accuracy, and on a language unseen by both encoder and decoder the
> choice of variant is immaterial (1.88 CER spread against 0.64–1.65 run noise).

That is a negative result with a measured power bound, an orthogonal sample-efficiency finding,
and an OOD boundary condition. It is publishable as a rigorous null; the same data presented as
a win is what a reviewer would take apart.
