# Summary: what was run, and what it found

The one-page index. `FINDINGS.md` is the long-form evidence, `OUTPUTS.md` the file provenance,
`CLAUDE_CHANGES.md` the chronology. Read this to decide what to focus on.

**Primary metric is CER everywhere**, including the baseline and accent contrasts. WER stays in
`t7_baselines.csv` as a secondary column.

## Experiments

| # | What was measured | Script | Output |
|---|---|---|---|
| 1 | Region-matched vs mismatched decoder, per language | `analyze_region_match.py` | `t2_region_match{,_stats}.csv` |
| 2 | Per-run curve stats: best/final CER, convergence in steps, epochs, audio hours | `analyze_sample_efficiency.py` | `t1_sample_efficiency.csv` |
| 3 | Is the convergence point set by language or decoder (ICC + permutation + noise floor) | `analyze_convergence_clustering.py` | `t10_convergence_{clustering,by_language}.csv` |
| 4 | Region-match effect vs training-stream size | `analyze_volume_interaction.py` | `t5_volume_{interaction,stats}.csv` |
| 5 | Treatment size: each decoder's post-training share of the target language | `analyze_exposure.py` | `t8_exposure{,_stats}.csv` |
| 6 | Overfitting and generalisation gap by tier / domain / accent | `analyze_loss_metrics.py` | `t6_loss_{metrics,by_axis}.csv` |
| 7 | Run-to-run noise floor from replicate pairs | `analyze_replicates.py` | `t9_replicate{s,_stats}.csv` |
| 8 | Kreol Seselwa: unsupported by both frozen components | `analyze_ood_crs.py` | `t3_crs_ood.csv` |
| 9 | Off-the-shelf baselines, and trained-minus-baseline over the same configs | `analyze_baselines.py` | `t7_{baselines,training_vs_baseline}.csv` |
| 10 | Held-out dialect transfer, and whether a cheap eval predicts it | `analyze_accent_transfer.py` | `t11_accent_{transfer,correlations,by_language}.csv` |
| 11 | Stream reconstruction from logged scalars | `analyze_data_accounting.py` | `t4_data_accounting{,_by_language}.csv` |
| 12 | Dataset/eval registries and audio statistics | `build_manifests.py` | `data/manifest_{training,eval_sets}.csv` |

Guards: `test_utils_port.py` (unit), `verify_paper_numbers.py` (140 numbers, 33 orderings),
`verify_eval_pairing.py`, `verify_interleave_semantics.py`, `verify_dataset_durations.py`.
`bash plotter.sh` rebuilds everything and runs all of them.

Population: serial 0 is the 12x4 grid; serial 2 controls; 1/3/4/5 superseded or replicate;
serial 10 baselines (127 finished); serial 11 trained checkpoints (158 finished, 43/43 cells).

## Findings, ranked by what a reviewer will find compelling

**1. The headline result is null, and it is bounded rather than empty.**
Region-matched decoder is 0.55 CER better on average, Wilcoxon p = 0.966, 6 of 11 languages
favour matched. Against a minimum detectable effect of 4.90 CER and a between-run sd of 1.17 CER
measured from 10 replicate pairs. The manipulation is ~1.3 percentage points of post-training
data. A null with a measured floor and a measured treatment size is publishable; an unbounded
one is not. — `t2_region_match_stats.csv`, `t9_replicate_stats.csv`, `t8_exposure.csv`

**2. Projector training pays off exactly on the underrepresented languages.**
15 of 43 cells beat the best off-the-shelf model; the benefit tracks how badly that model was
already doing (Spearman rho = -0.43, p = 0.004). Every large win is a thin-coverage language:
Amharic -102.6 CER in-domain and -78.7 on FLEURS, Kreol Seselwa -68.8, Hausa -45.4, Tanzanian
Swahili -32.0. Below the median baseline CER of 15.8 it *costs* a median 3.7 CER. This is the
paper's low-resource argument, and it is the strongest positive result here.
— `t7_training_vs_baseline.csv`

**3. Kreol Seselwa is the cleanest single case.**
Supported by neither the frozen encoder nor any decoder. Whisper-Medium and Voxtral-Mini **fail
to produce output at all**; Qwen2-Audio-7B manages 85.18 CER. A trained projector reaches 16.41.
The four LisTAya variants land within 1.99 CER of one another (17.92-19.91), so which regional
variant is paired with it does not matter here either. Counting the non-regional Tiny Aya base
control as a fifth widens the spread to 3.43 -- state which set is meant. — `t3_crs_ood.csv`, `t7_baselines.csv`

**4. Dialect transfer fails completely: 0 wins in 18 held-out varieties**, median cost 4.1 CER.
The two apparent wins in the wider set were Tanzanian Swahili and Indian Urdu — both **in their
cells' training streams**, because `in_domain_role == 'accent_transfer'` means "not the primary
point", not "held out". A reviewer would have found this; catching it is worth stating.
— `t11_accent_by_language.csv`, `utils.is_trained_variety`

**5. Overfitting falls monotonically with resource tier.**
Eval-loss rise 0.211 / 0.188 / 0.041 / 0.003 and fraction-of-run-to-best 0.282 / 0.292 / 0.627 /
0.816, across very-low / low / mid / high. Medians of per-language medians — the per-run
aggregation does **not** show it, and the paper must say which it uses. — `t6_loss_by_axis.csv`

**6. English inverts the design, and exposure predicts the inversion.**
The Fire mix is 46.2% English against Water's 17.0%, so English is the one language whose
*matched* decoder saw 15.2 pp *less* of it. English is also the single worst matched cell
(+7.39 CER). Where label and exposure disagree, the outcome follows exposure. Across all eleven,
r = -0.53 (p = 0.093) — directionally right, underpowered. — `t8_exposure.csv`

**7. Convergence is nearly language-independent.**
Every one of the 48 runs converges within 90-350 optimiser steps (median 170) across corpora
spanning 75x in size. Language explains only 31% of the variance in log-steps (F = 2.81,
p = 0.0095; permutation p = 0.0083); the median language spans 1.59x across its four decoders
against a 1.18x replicate noise floor. Measured in *epochs* the ICC is 0.94, which is the
corpus-size denominator, not a finding. — `t10_convergence_clustering.csv`

**8. A cheap in-domain eval partially predicts accent robustness; FLEURS does not.**
In-domain CER predicts held-out-variety CER at rho = 0.42 (p = 0.001, n = 54) and survives
removing the language mean at rho = 0.30 (p = 0.026). FLEURS: 0.20 -> 0.06 (p = 0.66). Most of
any pooled correlation is the language, not the model (0.53 -> 0.15 on the full set).
Practical, and the pooled/centred gap is itself the caution. — `t11_accent_correlations.csv`

**9. A secondary hypothesis formed and died — report it as such.**
"The region-match effect decays with training volume" held at rho = 0.96 on 7 languages and fell
to 0.48 (p = 0.13) at 11. Training volume and baseline difficulty are collinear (rho = -0.82,
p = 0.002); the partial correlation is -0.13 (p = 0.72), inconclusive rather than null.
Reporting a died hypothesis is a strength if it is framed as one. — `t5_volume_stats.csv`

**10. Learning speed and endpoint accuracy are different axes.** Convergence rank Fire 2.00,
Water 2.08, Earth 2.75, Global 3.17; accuracy rank Earth 2.17, Fire 2.50, Water/Global 2.67.
Both place Global last — the only hint that regional specialization does anything. One seed per
cell: suggestive, not established. — `t1_sample_efficiency.csv`

## Known weak points

- One seed on most cells; 10 of 52 runs have a replicate.
- Ten of twelve languages train on WorldSpeech and evaluate on FLEURS, so decoder choice is
  entangled with a domain shift. In-domain results are reported alongside but do not separate.
- Two in-domain languages only, so the generalisation-gap contrast (0.184 vs 0.179) is
  underpowered in either direction and no conclusion is drawn from it.
- `en_pk` is degenerate rather than merely bad (93.87 -> 378.71 WER); flagged, not dropped.
- Whether decoder specialization affects downstream spoken-language *understanding* — where the
  decoder's own multilingual knowledge is exercised — is untested. ASR only.
