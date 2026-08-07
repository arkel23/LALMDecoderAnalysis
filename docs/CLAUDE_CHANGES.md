# Change log

## 2026-08-07 (fifth) — CER everywhere, and a one-page summary

**The baseline and accent work was in WER while the rest of the study is in CER.** Serial 0
selects on `eval/cer` and every serial-0 statistic is CER, but `analyze_baselines.py` defaulted
to `--metric wer` and the accent analysis followed it. CER is complete on both eval serials
(127/127 and 158/158), so the switch cost no coverage.

Switched: `analyze_baselines.py` default metric, `analyze_accent_transfer.py` columns
(`variety_cer`, `primary_cer`, `fleurs_cer`), the three table generators and their captions,
`verify_paper_numbers.py`, and every affected number in `main.tex`. Only two WER mentions remain
and both are deliberate: Fong et al.'s own reported figures, and one labelled review comment.

The CER framing is also a better story. Training beats the best off-the-shelf model on 15 of 43
cells and the benefit tracks the baseline's weakness (rho = -0.43, p = 0.004), with every large
win on a thin-coverage language: Amharic -102.6 in-domain and -78.7 on FLEURS, Kreol Seselwa
-68.8, Hausa -45.4, Tanzanian Swahili -32.0. Split at the median baseline CER of 15.8, it wins
12 of 22 above and 3 of 21 below.

**A real error surfaced while verifying the summary.** The paper said "the four decoders span
17.92--21.35 CER, a range of 3.43" on crs_sc. 21.35 is `tiny-aya-base`, a non-regional CONTROL
and a fifth model. The four LisTAya variants span 17.92-19.91, a range of **1.99**. 3.43 is the
five Tiny Aya models, 3.96 all six. `FINDINGS.md` had said this correctly; the paper lost it.
Three derived checks now pin the four-variant figures.

**New `docs/SUMMARY.md`**, and README points at it. One page: a table of every experiment with
its script and outputs, then the findings ranked by how compelling they are to a reviewer, with
the CSV behind each. Every number in it was re-derived from its CSV before committing.

## 2026-08-07 (fourth) — an absent baseline cell is three different things

Asked how `crs_sc` is handled when Whisper-Medium and Voxtral-Mini fail on it. The answer was
**NaN, undifferentiated**: the failed rows are kept in `t7_baselines.csv` with `state='failed'`
and NaN `wer`/`cer`, the paper table printed `--`, and `coverage()` reported "5 cells not
finished ... aggregates cover ONLY what has finished". Three of those five can never be run and
two failed structurally, so the line read as "come back later" when nothing will change.

`coverage()` now splits them, deriving the first from the eval registry:

- **not_applicable** (3) --- FLEURS has no Seychellois Creole, so `crs_sc`/cross-domain has no
  config for any model. It can never be run.
- **failed** (2) --- `crs_sc` in-domain for Whisper-Medium and Voxtral-Mini. The run exists and
  failed because the models do not accept the language. That is the paper's `crs_sc` argument,
  not a gap.
- **pending** (0) --- the only kind that fills in on its own, and the only kind that now
  triggers the "aggregates cover ONLY what has finished" warning.

`tab_baselines_full` marks a failed cell `\textdagger` with the reason in the caption instead of
printing `--`, which had been indistinguishable from a pending cell. Failed rows are still kept
in the CSV; only `compare_with_trained` and the aggregates filter to finished, so `crs_sc`'s
best baseline is Qwen2-Audio-7B --- the only one that ran.

**Serial 11 completed at 43 of 43 cells during this pass.** Training now wins 15, the crossover
strengthens to rho = -0.46 (p = 0.002), the largest gain is 81.7 WER, and held-out dialect
transfer is 0 of 18.

## 2026-08-07 (later still) — early stopping, described for QuantizedASR but not applied

`for_quantizedasr/qasr/train/add_early_stopping.py`, mirroring the target path. QuantizedASR is
untouched; the file states the diff and carries the two functions to paste.

Read against the live `build_trainer`: `Seq2SeqTrainingArguments` already sets every field the
callback needs, `weight_decay=args.weight_decay` is already wired (so weight decay is a CLI
value, not a code change), and `DecoderASRTrainer.__init__` already accepts and forwards
`callbacks`. The only gap is that neither construction site passes it.

Four traps, each of which disables or defers stopping with a warning rather than an error, so
`validate_early_stopping` turns them into construction-time failures:
`save_steps != eval_steps` defers the stop; `load_best_model_at_end=False` leaves
`state.best_metric` unset (and its CLI flag is `store_false`, so passing it is what turns it
OFF); `greater_is_better` is hardcoded False in `build_trainer`, silently inverting for an
upward metric; `prediction_loss_only=True` computes no metrics, so anything but `loss` resolves
to None. All four exercised here against the pasteable function.

Caveat recorded for whoever runs it: `final_minus_best`, `late_sd` and the eval-loss rise
carrying the overfitting-by-tier result are all defined on a completed curve. Early-stopped runs
truncate exactly the post-minimum tail those measure, so they belong in a new serial rather than
in serial 0.

## 2026-08-07 (later) — accent transfer, and a variety set that was not held out

**`accent_transfer` was never a held-out set.** `in_domain_role` returns it for anything that is
not a cell's primary point, and `TRAIN_CONFIGS` trains `sw_ke` on `sw_ke + sw_tz`, `ur_pk` on
`ur_pk + ur_in`, `ha_ng` on `ha_ng + ha_td`. So two of the 20 evaluated varieties were training
data. Splitting on it changes the dialect result outright:

| | wins | median delta WER |
|---|---|---|
| genuinely held out | **0 / 14** | +6.0 |
| seen in training | 2 / 2 | -32.3 |

The paper had attributed those two wins to the weak-baseline effect. They are seen data.
Corrected, along with the "20 held-out varieties" count (18 held out, 2 trained).

`utils.TRAINED_VARIETIES` is **derived from `TRAIN_CONFIGS`**, not typed — a typed list is what
let the two eval sweeps drift once already. `is_trained_variety` plus six unit tests.

**New: `analyze_accent_transfer.py`.** One row per (model, variety) joined to that same model's
in-domain and FLEURS anchors, `validate='many_to_one'` on both. Writes `t11_accent_transfer`,
`t11_accent_correlations` and `t11_accent_by_language`.

**The language main effect is most of any pooled correlation.** Over all varieties, in-domain
WER predicts variety WER at rho 0.56; centre both axes within language and it is 0.12
(p = 0.38). On the held-out population in-domain predicts at 0.44 pooled and **survives**
centring at 0.29 (p = 0.035), while FLEURS does not (0.25 -> 0.20, p = 0.15). Every correlation
is emitted at all three aggregations rather than the flattering one, and an ordering check pins
that centring must shrink the pooled figure — negative-tested.

**Hausa's asymmetry is now encoded, not lore.** Its in-domain point is WorldSpeech `ha_td` and
its out-of-domain point is FLEURS `ha_ng`; WorldSpeech `ha_ng` test is the selection split and
excluded, because the FLEURS Hausa validation split was too slow to evaluate. `ha_ng` and
`crs_sc` are the two cells not selected on FLEURS, so only theirs is zero-shot with respect to
selection — `utils.selected_on_fleurs`, carried as a column. Neither contributes a held-out
variety today, so it guards rather than moves the numbers.

**Serial 11 grew 33 -> 36 cells mid-run** and the checker caught two stale counts in the paper
before they were committed.

## 2026-08-07 — convergence in steps and epochs, and the baseline contrast inverts

**t1 gained step and epoch columns.** It recorded the convergence point only in audio hours.
`train/global_step` and `train/epoch` are on every eval row and `curve_stats` already had the
index, so `step_to_best`, `step_to_{1.25,1.5,2}x_best`, `step_total`, `epoch_at_*` and
`epoch_total` cost two lines each.

**New: `analyze_convergence_clustering.py`.** Is the convergence point a property of the
language or of the decoder? A one-way variance decomposition of log10(convergence point)
grouped by language, with the F test, a 10,000-shuffle permutation null, and a noise floor from
the replicate serials via the same `curve_stats`.

Every run converges within **90-350 steps** (median 170) across corpora spanning 75x in size.
Language explains **31 %** of the variance in log10 steps (F = 2.81, p = 0.0095; permutation
p = 0.0083). The median language spans 1.59x across its four decoders against 1.18x between two
runs of one configuration (10 pairs, one-sided p = 0.006) — real, but small. Tamil, the example
that prompted the question, is 120-290 steps: one of the *widest*, not the tightest.

Measured in epochs the ICC is 0.94, which is not a finding: epoch is
`step * effective_batch / stream size` to r = 0.998 and stream size is a language property by
construction. `t10` carries `icc_is_tautological` so the 0.94 cannot be quoted as clustering.

**Serial 11 went from 4 to 33 evaluated cells and inverted a headline claim.** The ordering
check added yesterday — "training beats the best baseline on every cell" — went red. It is now
13 of 33, and *which* 13 is the result: the benefit tracks the baseline's weakness
(Spearman rho = -0.45, p = 0.009). Where the baseline exceeds 50 WER the projector wins 6 of 9
(median gain 25.0, largest 77.1); under 20 WER it wins 1 of 10 and costs a median of 5.0.

The same data gives a dialect result. On its own training variety the projector beats the
baseline on 7 of 10 cells (median gain 5.1 WER); on held-out varieties of the same languages,
2 of 13 (median cost 5.6 WER). Both wins are cells whose baseline was already above 45 WER, so
they are the weak-baseline effect rather than an exception.

`main.tex` was corrected in three places rather than added to — the previous claim was false
against the new data, not merely incomplete. One `% REVIEW:` flags a degenerate pair
(en_pk: 93.87 to 378.71 WER) and one `% TODO` records that the sweep is at 33 of 43 cells.

## 2026-08-06 (second pass) — figures, discussion, and the paper enters the number checker

**Additive again.** Nothing was removed from `main.tex`. Added: a `Related Work` section, six
Results subsections (exposure, volume interaction, learning speed, the domain-gap null, the full
baseline matrix, accent transfer), `Discussion`, `Conclusion`, three generated tables, and the
three existing figures — the first draft used none of them. 6 pages to 11.

**Three new generated tables.** `tab_exposure`, `tab_baselines_full` and `tab_dataset_stats`.
The trap: in a Python raw string `r'\\textbf'` is a literal double backslash, so the four
earlier table headers (`\textbf`) were right and the three new ones silently emitted a LaTeX
line break before every column name — 101 errors from three characters. Same for `\\Delta`.

**The paper is now scanned by `verify_paper_numbers.py`.** `DOCS` gained `main.tex`, guarded on
existence so a bare clone still passes. Eleven derived checks and four orderings were added for
the numbers the new sections print: the exposure correlation, English's negative excess, the
largest gain training buys, the held-out variety count, the four Whisper hour figures, and the
per-variant mean ranks on both convergence speed and endpoint accuracy. 128 numbers, 28
orderings, 102/128 high-specificity. Negative-tested by flipping the sign of English's
`excess_pp`: one ordering broke, one number failed, exit 1; restored, exit 0.

It immediately caught one drift: the Conclusion printed "up to 53 WER" against a CSV value of
-52.83. The paper now prints 52.8.

**Two figure titles asserted more than the data does.** The volume figure was titled
"Region-matched decoder benefit decays with training-data volume" while the text reports
rho 0.48, p 0.13 — retitled to "Region-match effect against training-data volume". The crs_sc
figure advertised "5 finished TinyAya variants; Qwen3-4B control still training", but serial 0
holds only the four LisTAya variants and Qwen3 is out of the paper's scope.

**`t7_baselines` uniqueness now scopes to finished runs.** A failed `fr_ca`/Voxtral eval sitting
beside its in-flight retry tripped the assert. A re-run is not a duplicate eval; the constraint
belongs on the rows that carry results, matching the `is_canonical` convention used elsewhere.

**One prose error caught by re-derivation, not by the checker.** The accent-transfer sentence
listed 7 English + 8 Spanish + 2 French and then said "20 held-out varieties in all". There are
20, but three of them are single varieties of Swahili, Tamil and Urdu that the list omitted.

**One `% REVIEW:` comment added.** The Whisper hour figures quoted in the baselines section are
ASR plus translation hours; for the low-resource entries almost all of it is translation
(Marathi: 0.6 h ASR of 289).


## 2026-08-06 (later still) — first draft of the paper's Results

**Venue.** ROCLING, calibrated from proceedings rather than instructions: 149 papers across
2023-2025 have a **median of 8 pages including references**, 60 % at 8 or fewer, 32 % at 7 or
fewer. The draft currently compiles to **6 pages** with 0 errors, 0 undefined citations and 0
overfull boxes.

**Additive only, as instructed.** Nothing was removed from `main.tex`. The `PH` placeholders
under Results and Limitations were filled, a related-work paragraph was appended before Methods,
and six `% REVIEW:` comments flag things for manual attention rather than editing them: the
Indonesian (`in_in`) and Amharic (`am_te`) training configs, Urdu's pooling order, Tamil now
being single-config, the intro's "10 languages" against the abstract's 12, and the option of
replacing the hand-written Table 1 with the generated one.

**Tables are generated.** `build_paper_tables.py` writes `ACL26_LALMDecoder/tables/*.tex` from
the manifests and results CSVs, wired into `plotter.sh`. Two LaTeX traps cost time and are worth
recording:

- `\input` of a bare tabular body **does not work** — inside an alignment TeX cannot read a file
  boundary and the following `\bottomrule` fails as a misplaced `\noalign`. The generator now
  emits the complete float, caption and label included.
- The last row **must** keep its trailing `\\`, or the row is still open when `\bottomrule`
  expands. An earlier "fix" that stripped it reintroduced the same error.
- A 6-column table silently **overlapped its neighbour** while LaTeX reported no error. The wide
  tables are now `table*`. Caught only by rendering the page, which is why the convention exists.

**Serial 11 arrived.** 10 finished runs, and the first training-vs-baseline contrast:
Hausa FLEURS 95.56 -> 68.72, Swahili FLEURS 51.52 -> 35.02, Swahili WorldSpeech 25.93 -> 20.85,
sw_tz 78.10 -> 25.27. Training beats the strongest downloadable model in 4 of 4 cells so far.
`compare_with_trained()` had been handed the raw frame while expecting the built one, and raised
the first time serial 11 was non-empty.

**Language coverage as a contribution.** Of the 12 study languages, seven — French, Marathi,
Urdu, Swahili, Hausa, Amharic and Kreol Seselwa — are named nowhere in the 38 cited works. The
claim in the paper is scoped to the speech-LLM literature surveyed, since African ASR reviews do
cover Amharic, Hausa and Swahili; Kreol Seselwa appears unstudied in this architecture entirely.

**Docs.** `HANDOVER.md` -> `README.md` (it is the index a reader looks for).
`docs/RELATED_WORK.md` retired to `deprecated/` now its anchors are in the paper.
`docs/FINDINGS.md` lost the framing sections that predated the paper.

## 2026-08-06 (later) — ta_lk removed from the record; one bibliography

**Verified over the full config, not a sample.** Reading the `duration` column for all 23,261
`ta_lk` rows (casting the Audio column with `decode=False`, so nothing is decoded) gives a single
unique value: **30.000000**. The upstream filter keeps a clip only when
`length < max_input_length`, so all 23,261 fail and Tamil's stream is `ta_in` alone. An earlier
attempt to confirm this by decoding every clip was OOM-killed at 29 % on this 7 GB box; the
metadata read settles it in seconds and covers every row rather than a 100-row sample.

`purge_ta_lk_from_configs.py` then removed `ta_lk` from `dataset_train`, and the aligned entry
from `dataset_path_train` and `split_train`, on the **6** runs that carried it (4 in serial 0,
2 in serial 2). Dry-run first; W&B now reports 0 runs carrying it.

`utils.py` follows: Tamil is a single-config stream, out of `MULTI_CONFIG_TRAIN`, out of
`ACCENT_MATCH` (it trains and evaluates the same variety), and `KNOWN_AT_CAP_CONFIGS` is empty.
`ta_lk` stays in `TRAIN_CONFIG_TO_CELL` — that map normalises *any* config to its study cell, and
removing it silently detached the eval set from Tamil.

**No substantive number moved.** Only the pre-filter bookkeeping columns changed —
`expected_stream_examples` 32,107 -> 8,846 and `n_dropped_by_cap` 23,261 -> 0, plus the ratios
derived from them. `stream_post_filter` is still 8,846, Tamil's delta still -14.70, and t1, t2
stats and t8 are byte-identical. Two ordering claims that asserted the loss were replaced by
claims that every language now reconciles.

`ta_lk` remains an eval set, described plainly as a Tamil variety outside the training mix —
the same axis as the English and Spanish variants.

### Bibliography

`LisTAya_updated.bib` is the single bibliography; `LisTAya.bib` and the ACL template's
`custom.bib` moved to `deprecated/`. Nothing was lost: the only key unique to `LisTAya.bib` was
`zhang-etal-2026-language`, the ACL Anthology version of a paper `LisTAya_updated.bib` already
carries as `zhangLanguageFamilyMatters2026` (arXiv). **Suggestion for a later pass, not applied:**
the Anthology entry has better metadata than the arXiv one now cited, and
`imamAutomaticSpeechRecognition` carries no year or venue.

Added two references the survey turned up: `imamAutomaticSpeechRecognitionAfrican2025`
(systematic review of ASR for African low-resource languages) and `adjoviSpeechTextCorpora2026`
(ASR data acquisition for Fongbe and Hausa). 38 entries.

### Venue calibration for the paper

149 ROCLING papers, 2023-2025: **median 8 pages including references**, 60 % at 8 or fewer, 32 %
at 7 or fewer. So 8 pages total is the norm and shorter is unremarkable — with ~38 references
costing 1.5-2 pages, the content budget is roughly 6.

## 2026-08-06 — one registry for the eval datasets, and the source-of-truth manifests

`worldspeech_ha_ng_test` is Hausa's training-time selection split. That was found, recorded in
`utils.SELECTION_SPLIT`, and used to move Hausa's in-domain point to `ha_td` -- and then the
config was left in both sweeps anyway.

**The lists had already drifted.** The deployed `eval_lalm_baselines.sh` carried 43 datasets with
`ha_ng` removed by hand; `eval_lalm_decoder_txf.sh` carried 44 with it in. So it was excluded
from the baselines, where it is harmless, and included in the trained sweep, where it is
contaminated -- exactly backwards. `missing_runs.py` had a third copy of the list and reported
three phantom `ha_ng` runs as missing.

### The registry

`for_quantizedasr/tools/preprocess/eval_datasets.csv`, written by the WorldSpeech generator.
Both sweeps read it at runtime (header-driven awk) and `missing_runs.py` reads it too, so there
is one list. `use_in_sweep` is **derived** -- `not is_selection_split(...)` -- so the exclusion is
computed, never typed. It flags exactly one config across all 44.

A real trap on the way: `csv.DictWriter` defaults to CRLF, which left a trailing `\r` on the last
field and made every shell reader match nothing. The writer now sets `lineterminator='\n'` and
the awk strips `\r` anyway.

### Audit: most of what was "missing" already existed

- **dataset statistics** -- every eval run already logs `num_samples` and
  `audio_length_s_{mean,std,min,max}`. Complete for 43/43 eval sets. Only an extractor was needed.
- **TinyAya languages and regions** -- already in
  `data/tinyaya_report/tinyaya_language_composition.csv`: 69 languages with `report_region`.
- **train/test record** -- already in the wandb run configs. Now extracted.
- **Whisper hours** -- copied as a frozen snapshot from MultilingualQASR. 11 of our 12;
  `crs_sc` is absent because Whisper does not support it, which is the finding, not a gap.

`build_manifests.py` writes `manifest_training.csv` (12 cells, with the column that names which
eval config collides with a cell's selection split -- populated for `ha_ng` alone),
`manifest_eval_sets.csv` (registry + audio statistics) and `language_hours_whisper.csv`.

### Numbers

| | before | after |
|---|---|---|
| serial 11 | 176 (44/variant) | **172** (43/variant) |
| serial 10 expected | 132 | **129** |
| serial 10 missing | 9 | **6** -- the three ha_ng entries were never real |

The training analysis is untouched: t1, t2, t3, t5, t6, t8 and t9 are byte-identical.

### Also fixed

`verify_eval_pairing.py` printed `[FAIL]` and exited 0 -- the same defect that once let
`plotter.sh` report success over a failing guard. It now records failures and exits non-zero,
negative-tested.

## 2026-08-05 (later still) — missing-run check, and what serial 10 is actually missing

`missing_runs.py`, adapted from ChineseQASR's `missing_scripts.py` (same finished-wins state
lookup, same MISSING / FAILED / CRASHED rerun rule, no quantisation axis). The serial-10 grid is
an explicit cross product held in the script; `test_utils_port.py` pins both lists against
`eval_lalm_baselines.sh` so the duplication cannot rot. Serial 11 is not a cross product, so it
takes `--pairings`.

**The baseline roster is three models**, not six: whisper-medium, which is the study's own
encoder and so the like-for-like reference, plus the two commercial LALMs. whisper-tiny, -small
and -large-v3-turbo are dropped -- they answer a Whisper-scaling question this study is not
asking, and they had never been run.

**Serial 10: 132 expected, 123 done, 9 to run.**

| model | missing |
|---|---|
| `voxtral_mini_3b` | 4 |
| `qwen_2_audio_7b` | 3 |
| `whisper_medium` | 2 |

The nine stragglers are all WorldSpeech: `crs_sc` and `ha_ng` for whisper-medium, those plus
`fr_ca`/`fr_cd` for Voxtral, and `ha_ng`/`hi_in`/`mr_in` for Qwen2-Audio.

**The sweep named the wrong Qwen.** `MODEL_CONFIGS` listed `qwen_2_audio_7b_instruct.yaml` while
every actual run used `qwen_2_audio_7b.yaml`, so 41 finished runs matched no expected cell and
the instruct variant showed as 44 missing. The instruct variant performs badly and is not wanted,
so the sweep now names the plain one and a test asserts neither list mentions "instruct".

## 2026-08-05 (later still) — both sweeps request WER and CER explicitly

`--eval_metrics` defaults to `['wer_all']` upstream (`qasr/misc/misc_utils.py:158`), so CER was
not computed unless the dataset YAML asked for it. Both sweeps now pass
`--eval_metrics wer_all cer`, which makes it independent of how a config happens to be written.

That exposed a second gap: serial 10 already carried CER for 123 of its 129 runs, but
`analyze_baselines.py`'s `WER_FAMILY` did not list `cer`, so `t7_baselines.csv` dropped the
column. CER is the metric the training runs report, so it is what makes serial 11 comparable to
them. Added.

## 2026-08-05 (later still) — the txf sweep evaluates best checkpoints only, four variants only

The sweep swept all 104 generated configs: both checkpoints per cell and both control decoders.
Step 1000 is the overfitted end of the curve, and the best-to-final gap is already read off the
training curves in t1, so evaluating it buys nothing. The controls exist for 2 of 13 languages
and are not part of the grid.

The generator now emits **only what the sweep needs** -- 48 configs, the best checkpoint of each
of the four grid variants across 12 languages -- so the sweep's glob matches exactly one file per
(variant, language). `INCLUDE_FINAL` and `INCLUDE_CONTROLS` flip that back if the step-1000 or
control configs are ever wanted. Skip counts are now reported by reason and reconcile exactly:
48 written + 4 superseded lang + 52 final step + 4 control decoder = 108 Hub checkpoints.

| | before | after |
|---|---|---|
| model configs generated | 104 | **48** |
| evaluations | 368 | **176** (44 per variant) |

`--eval_set` is gone. It defaulted to `primary`, so a plain `--models fire` ran 26 of the 44
datasets -- the flag's default contradicted the stated target and that is exactly how it was
caught, by eye, in a real run. The sweep now always evaluates all 44, and
`verify_eval_pairing.py` asserts it consumes every pairing field so the subset cannot come back.
The guard already reported "44 pairings"; it simply was not checking that the sweep used them.

`--models` now selects among `{earth, fire, global, water}` by short name and rejects anything
else, so `--models fire` runs that variant's 44 evals and nothing else. It was previously a list
of decoder slugs that happened to work but read as unused.

The sweep keeps a `_1k` guard even though the generator no longer emits those configs: an earlier
run of the generator wrote 104 files, and if they are already sitting in a QuantizedASR checkout
the sweep must still ignore them. Verified by dropping a stray `_1k` config in and confirming the
count stays at 176.

## 2026-08-05 (later still) — serial 0 becomes the 12x4 grid, and two claims do not survive it

Serial 0 held four different kinds of run: the grid, two control arms, and (once the `es_es` runs
were re-added from another project) a superseded condition. Every cross-language aggregate had to
remember to filter, and that filter had failed twice. Now a serial means one thing.

| serial | n | contents |
|---|---|---|
| **0** | 48 | the grid: 12 languages x {earth, fire, global, water} |
| 1 | 8 | superseded grid re-runs (`am_et`, `crs_sc`, seed 42) |
| 2 | 4 | control arms: `base`, `qwen3-4b` |
| 3 | 1 | superseded control (`crs_sc`/`base`) |
| 4 | 1 | same-seed replicate (`en_us`/`water`) |
| 5 | 4 | superseded `es_es` condition |

### The trap that had not fired yet

The re-added `es_es` runs share `dataset: es_419` with the `es_mx` runs. `mark_canonical` breaks
ties by **earliest**, and `es_es` is from 2026-07-17 against `es_mx`'s 2026-07-30 — so the next
`bash plotter.sh` would have silently made every Spanish number come from the superseded
condition. The earliest-wins rule is right for a half-trained re-run and wrong here, and no guard
caught it. Recorded in `CLAUDE.md`; the fix is the serial boundary, not another filter.

### Two claims did not survive the clean grid

**The overfitting-by-tier result survives, but only per-language.** `analyze_loss_metrics.py`
had no variant filter, so `ta_in`'s `base` and `qwen3-4b` sat in the `very_low` tier.

| | very_low | low | mid | high | monotone |
|---|---|---|---|---|---|
| was published (per-run, contaminated) | 0.195 | 0.188 | 0.028 | 0.003 | yes |
| per-run, clean grid | 0.175 | 0.188 | 0.028 | 0.003 | **no** |
| **now published (per-language, clean)** | **0.211** | **0.188** | **0.041** | **0.003** | yes |

t6 now reports medians of per-language medians and carries an `aggregation` column.

**The domain-shift claim does not survive at all.** Earlier drafts reported a ~6x larger
generalisation gap cross-domain (0.196 vs 0.032). On the clean grid the two domains are
indistinguishable: **0.184 vs 0.179** per-language, 0.190 vs 0.173 per-run. The 6x was an
artifact of the control arms — `crs_sc`'s six near-zero runs outnumbered `ha_ng`'s four in the
in-domain pool. Only two languages evaluate in-domain, so the comparison is underpowered either
way. The ordering claim that asserted the gap was larger is replaced by one pinning it as a null,
so it cannot silently become a claim again.

### Coverage gap this exposed

The tier ordering was machine-checked but the four **values** were not, so they went stale while
`verify_paper_numbers.py` stayed green. Same for the domain gaps. Both are now in `DERIVED`:
**109 numbers checked, up from 97.**

### Unchanged, as predicted

Region-match stats (mean -0.55, p 0.966, MDE 4.90), t5 volume, t8 exposure and t1's grid
statistics (median `late_sd` 1.09, mean `final_minus_best` 5.36) are byte-identical — they
already filtered to `CORE_VARIANTS`.

t9 still finds **10 pairs, between-run sd 1.174**: replicate pairing now runs over
`REPLICATE_SERIAL_PAIRS = ((0,1), (2,3), (0,4))`. Pairing 0<->1 alone would have given 8 pairs and
dropped the only `same_seed` one.

## 2026-08-05 (later) — re-runs migrated, and the 5x noise-floor claim does not survive

### Serial migration

The `am_et` and `crs_sc` re-runs finished and their checkpoints are on HF, so the 9 originals
(4 am_et: earth/fire/global/water; 5 crs_sc: those plus base — all seed 42) moved to serial 1.
Serial 0 is now 52 runs with **zero duplicates**: 4 variants per language, 6 for `crs_sc` and
`ta_in`. Serial 1 holds 10.

### The noise floor is measured, and the earlier claim is refuted

With 10 replicate pairs instead of 1:

| | value |
|---|---|
| between-run sd, all 10 pairs | **1.17 CER** |
| between-run sd, the 9 seed-varying pairs | **0.90 CER** |
| median within-run `late_sd` | **1.09 CER** |

An earlier entry concluded, from the single `en_us`/water pair differing by 5.01 CER, that
`late_sd` understated run-to-run variance by ~5x and that every uncertainty statement resting on
it was optimistic by that factor. **Nine further pairs do not support that.** They average 1.51
CER apart with a maximum of 3.21, so `en_us`/water is the outlier. Between-run sd is 1.07x the
within-run proxy, not 5x. `late_sd` is a defensible noise floor after all, and the minimum
detectable effect stands where it was.

Nine of the ten pairs vary the seed (42 vs 420), so they estimate seed sensitivity, not just
nondeterminism — the larger quantity, and it is *smaller* than the same-seed outlier.

### Numbers that moved

41 checked values changed, because `am_et` and `crs_sc` now report their re-runs. The
significant ones: Wilcoxon p on the primary contrast 1.000 -> **0.966**; matched-vs-global
p 0.638 -> **0.966** and mean delta -0.72 -> **-0.50**; `am_et`'s delta +0.64 -> **+1.32**; the
volume correlation 0.555/0.0767 -> **0.482/0.1334**, so it is further from significance; mean
final-minus-best 5.02 -> **5.36**. `crs_sc`'s best TinyAya variant improved to 17.92, narrowing
the all-six spread to 3.96 while Qwen3-4B still leads at 17.39.

The per-language table in `FINDINGS.md` is now regenerated from `t5` with half-up rounding
rather than hand-patched — an intermediate attempt used Python's banker's default and produced
5.01/6.13/-0.14 where the convention requires 5.02/6.14/-0.15.

### A checkpoint family was being silently dropped

`create_yamls_models_lalm_txf.py`'s `CKPT_RE` matched only `CohereLabs_tiny-aya-*`, so the four
**Qwen3-4B** checkpoints never became configs. The regex now matches any `org_name` decoder and
derives the filename slug from it, which leaves every existing TinyAya filename byte-identical
and adds `cq2a_whisper_medium_qwen3_4b_txf_*`. The sweep's glob hardcoded `tiny_aya` too;
`VARIANTS` now carries full decoder slugs.

Verified against HF: **108 repos, 108 matched**; 104 configs generated (108 minus the 4
superseded `es_es`), **104 of 104 reachable** by the sweep's glob. 13 languages for each of
earth/fire/global/water, 2 each for base and qwen3-4b.

## 2026-08-05 — comment cleanup and a documentation index

### Code: 24% prose -> 17%

449 lines removed, 469 of them comment or docstring. `utils.py` alone went 698 -> 566. What went
was narrative -- "an earlier version said X, that was wrong", "this replaced an earlier
snapshot", "did not survive replication", incident retellings. What stayed is one- or two-line
invariants. Every module docstring over 25 lines was rewritten to 5-15.

**The cleanup is provably output-neutral.** It was done with two helpers that refuse to touch
code: a comment-block replacer that asserts every deleted line starts with `#`, and a docstring
replacer that asserts the AST below the docstring is unchanged. 22 of 23 generated files came
back byte-identical; `t7_baselines.csv` differed only because serial 10 had grown from 16 to 84
runs mid-session, and regenerating it from the committed raw data reproduced the committed CSV
exactly.

One mistake worth recording: an early attempt replaced comment blocks by line range between a
header and the next code anchor, which swallowed `TINYAYA_COMPOSITION_CSV` and a dict definition.
Caught by an import failure. `utils.py` was reverted to HEAD and redone with the assert-guarded
helper, which cannot do that.

### Two real defects the cleanup surfaced

- **32 of 84 baseline rows had no `study_cell`.** `TRAIN_CONFIG_TO_CELL` covered only training
  configs, so the 16 held-out variants (`en_au`, `es_ar`, `fr_cd`, ...) carried no
  `language_name` and no `resource_tier` and dropped out of any grouped table -- which would
  have made the accent-transfer axis unreadable per language. Now mapped, with a test asserting
  every evaluated variant resolves to a cell and exactly one per cell is the in-domain primary.
  Accent transfer now groups correctly: 7 English, 8 Spanish, 2 French.
- **`NUMBER_PROVENANCE.md` and `REPRODUCE_FIGURES.md` were both stale.** The first quoted
  `rho = 0.964` and collinearity `-0.679` from the dead 7-language era (current: 0.555 and
  -0.818), missed t6-t9 entirely, and gave `final_minus_best` as 3.47 (now 5.30). The second
  said the faceted figure shows 9 languages and that the Qwen3-4B control was still training;
  it is 12 and it has finished. Neither was covered by `verify_paper_numbers.py`, which scans
  only the findings document.

### Docs: 9 files -> 8, with an index

`docs/OUTPUTS.md` merges the two stale files into one current table of every generated file --
what writes it, what it holds -- plus the definitions, the check-by-eye list, and the settled
items. That is the "crawl the generated files and data" entry point.

`HANDOVER.md` was rewritten as the index. It had opened with "**No data has been collected here
yet**" since 2026-07-30, against 57 training runs and a full analysis. It now carries current
state and a table pointing at each of the six docs.

`CLAUDE.md` lost its "Not yet applicable" section (`plotter.sh`, the tests and
`claude_process.sh` all exist now) and its claim that `SERIAL_DIC` is empty and the dicts are
placeholders. Its traps section now names the live ones: frozen snapshots, `is_canonical` over
`state == 'finished'`, and `ha_ng`'s selection-split collision.

## 2026-08-03 (later) — less is more: condense the code, consolidate the docs

Review verdict: too much prose, too little signal. Comments mixed necessary documentation with
narrative, so neither could be picked out. `CLAUDE.md` gains a "Less is more" section as the
standing rule.

### Code

| file | before | after |
|---|---|---|
| `create_yamls_worldspeech_lalm.py` | 245 | **58** |
| `create_yamls_models_lalm_txf.py` | 187 | **110** |
| `eval_lalm_decoder_txf.sh` | 143 | **87** |
| `verify_eval_pairing.py` | 178 | **71** |

The WorldSpeech generator now follows `create_yamls_short_ml.py`'s shape (one explicit list, one
loop) and emits **all 120** WorldSpeech configs from `configs/train/worldspeech_llama_questions.yaml`,
not just the study's 12. It no longer emits FLEURS configs -- QuantizedASR already ships 229, and
all 11 the sweep references exist -- nor a manifest, nor a pairings table. One script, one job.

`eval_lalm_decoder.sh` moved to `deprecated/`.

### Docs: 7 -> 5

| kept | holds |
|---|---|
| `FINDINGS.md` | what the study found + what to write (was `PLAN_ASSESSMENT` + `PAPER_ASSESSMENT`) |
| `RELATED_WORK.md` | external anchors only |
| `ROADMAP.md` | what to evaluate next + the augmentation study (was `EVAL_DATASET_PLAN` + `FUTURE_WORK_AUGMENTATION`) |
| `UPSTREAM_FIXES.md` | changes for QuantizedASR |
| `CLAUDE_CHANGES.md` | this chronology |

**The merge caught a contradiction.** `PLAN_ASSESSMENT.md` §8 still presented the volume-decay
hypothesis as the recommended framing -- "decays monotonically ... rho = 0.964, p = 0.0005",
nine languages, 4.10 CER spread, mean ranks 1.69/3.19 -- while §4.1 of the same document recorded
that hypothesis dying (rho 0.555, p 0.0767, not significant at n = 11). Section 8 is dropped;
`PAPER_ASSESSMENT.md`'s 12-language framing supersedes it.

`verify_paper_numbers.py` did not catch this, and could not: it asserts the correct value
*appears*, not that a wrong one is *absent*, so the stale figures sat beside the correct ones.
That limit is now recorded in the source. High-specificity passes rose 80 -> 83 once the
duplicates were gone.

**A second gap the merge exposed.** §5.3 quoted the `crs_sc` OOD spread as 1.88 across five
TinyAya variants while the CSV column reported 4.10. Both correct, different populations -- the
Qwen3-4B control has since finished. It reaches **17.39 CER, better than every TinyAya variant**,
which is a finding the document omitted entirely: on a language neither component has seen, what
matters is the decoder *family*, not its regional variant. Both aggregations are now stated.

Also added to §4.1: the partial correlation's p (**0.77**, 8 residual df, so inconclusive rather
than null) and the volume/difficulty collinearity (**rho -0.818, p 0.0021**).

## 2026-08-03 — literature positioning, generated eval pairing, and surviving a re-run window

### The re-run window broke the uniqueness contract, correctly

The five `crs_sc` re-runs (seed 420) started while the originals (seed 42) were still on serial
0, so every `crs_sc` cell held two runs and `assert_unique_keys` stopped the pipeline. That is
the guard doing its job -- the alternative is a paired subtraction silently seeing two rows per
cell, which is the cross-join defect the sibling repos shipped.

`analyze_sample_efficiency.py` now names a canonical run per `(model_id, dataset)`:

- prefer `finished` over unfinished; among equals prefer the **earliest**;
- every duplicate is **printed**, never silently dropped;
- `is_canonical` and `n_runs_in_cell` are columns, and `analyze_region_match.py` /
  `analyze_ood_crs.py` filter on them.

The rule is stability-first on purpose. A half-trained re-run (7 steps, CER > 100) must never
displace the 202-step run every existing number came from, and once it finishes the swap should
be a deliberate serial migration rather than something that happens on the next `bash
plotter.sh`. `state == 'finished'` alone was not enough: once both runs finish, both pass it.

### Median within-run `late_sd` was stale: 0.97 -> 1.06

The CSV has said 1.0649 since `ur_pk` landed. Three documents still printed 0.97, including the
noise-floor argument in `PAPER_ASSESSMENT.md`. Root cause: `verify_paper_numbers.py` scanned
only `PLAN_ASSESSMENT.md`, so the number was *in the spec* but the file printing it was never
read. It now scans `PAPER_ASSESSMENT.md` too (97 numbers, up from 94).

**Stated limit:** the check asserts the correct value *appears* somewhere. It does not assert a
wrong value is *absent*, so a stale figure beside a correct one still passes. Catching that
needs per-claim anchoring the specs do not carry.

### Unrecorded seeds get a sentinel, not NaN

Every run currently logs a seed (42 on the grid, 420 on the `crs_sc` re-runs), so this is
defensive. An unrecorded seed becomes `UNRECORDED_SEED = 0` plus a `seed_recorded` flag rather
than NaN, because `NaN != NaN` in pandas: a naive comparison calls every unrecorded pair "seed
varies" and pools a nondeterminism pair into the seed-sensitivity estimate, inflating it. The
sentinel is not guessed into `same_seed` either -- an unlogged run is not evidence it used 42.
`seed_status` is three-valued (`same_seed` / `seed_varies` / `unrecorded`), pinned by 5 unit
tests.

### The eval pairing is generated, not hand-written

`create_yamls_worldspeech_lalm.py` now emits `worldspeech_lalm_pairings.sh` from the same table
that writes the dataset YAMLs, and `eval_lalm_decoder_txf.sh` sources it. Verified identical to
the hand-written block it replaces (12 cells, 44 pairings, order aside) before swapping.

`verify_eval_pairing.py` checks the generated file and additionally fails if an inline
`PAIRINGS=(` reappears in the sweep. Negative-tested both ways: a `ur_pk` x `ha_td` injection
gives `[FAIL] no pairing crosses a language boundary -- model ur_pk (ur_pk) x data ha_td
(ha_ng)`, and a reintroduced inline block fails the source check. Both exit 1; restored, exit 0.

### Literature positioning

`docs/RELATED_WORK.md` records `arXiv:2508.05149`, a near-twin (frozen Whisper-large-v3-turbo +
frozen LLM + trained linear projector). Two anchors change how the results should be written:

- **SLAM-ASR needs 100-200 h** to match a Whisper-only baseline (10 h -> 14.0 WER, 100 h -> 7.6,
  200 h -> 6.4, vs Whisper 7.1). **Five of our twelve cells sit below it.**
- **Decoder choice matters enormously at low resource** -- EuroLLM 14.0 vs Salamandra 33.6 at
  10 h. Those are different model *families*; our variants differ by a median of 1.30 pp of
  post-training data. So our null is **consistent with** their positive result, and the claim
  becomes "specialisation at this magnitude does not measurably change ASR, and here is the
  magnitude".

They also use **no augmentation at all**, which is what makes
`docs/FUTURE_WORK_AUGMENTATION.md` a real gap. That document records the three arms (audio /
target / both), the correction that target-side augmentation with audio present *is*
backpropagable through a frozen decoder, the insertion point, and the blockers -- explicitly as
future work, not current work.

## 2026-08-02 (night) — evaluate every WorldSpeech variant, not just the trained one

The first version of `create_yamls_worldspeech_lalm.py` emitted one eval config per study cell
-- `ta_in` but not `ta_lk`, `ur_pk` but not `ur_in` -- reasoning that the eval should match the
reported cell. That was a poor trade: these are evaluation-only configs and WorldSpeech ships
several country variants per language, so covering them all costs inference time and nothing
else.

Now **33 configs**, tagged in a manifest CSV (kept out of the YAML so the configs stay exactly
the shape QuantizedASR's own generators produce):

| status | n | meaning |
|---|---|---|
| `in_training` | 15 | the variety the cell trained on |
| `held_out` | 17 | never in the training mix -- zero-shot accent/dialect transfer |
| `dropped_by_cap` | 1 | `ta_lk` |

`ta_lk` is the single most informative entry. Tamil's training config *names* it, but the strict
`< 30 s` cap removed all 23,261 of its clips, so the model never saw it. Evaluating there is
genuine held-out dialect transfer for a variety the config claims to have trained on.

The held-out English (7), Spanish (8) and French (2) variants are the **accent-robustness axis**
`EVAL_DATASET_PLAN.md` flagged as missing and proposed adding EdAcc for -- available in-domain,
on the training corpus, at no extra data cost.

The variant list was enumerated live from the Hub and then cross-checked against QuantizedASR's
`configs/train/worldspeech_llama_questions.yaml`, whose 120-entry `dataset_train` roster is the
frozen all-variants list. **They agree exactly for every study language.**

### Cost, made explicit

`eval_lalm_decoder_txf.sh` gains `--eval_set`:

| mode | evaluations | contents |
|---|---|---|
| `primary` (default) | **188** | FLEURS + in-training WorldSpeech |
| `all` | **334** | adds the 146 held-out-variant runs, including 10 on `ta_lk` |

---

## 2026-08-02 (evening, 2) — run-to-run variance is ~5x the noise floor I have been quoting

The `am_et` and `crs_sc` CHECKPOINTS were deleted from the training server. The wandb runs are
untouched -- all 4 `am_et` and 6 `crs_sc` runs are still `finished` with complete curves, so
every serial-0 analysis remains valid. What is lost is the ability to evaluate those cells in
serial 11, which forces a re-run.

That re-run is an opportunity, because of something the existing data already shows.

### Every run used seed 42, and a same-seed re-run still moved 5 CER

| en_us / water | best CER | final CER |
|---|---|---|
| serial 1 (first run, n4cot5v7) | 17.06 | 20.12 |
| serial 0 (re-run, pwnz2zno) | 12.05 | 13.12 |

Both `seed = 42`. The 5.01 CER gap is pure nondeterminism -- streaming order, GPU
non-associativity -- not seed sensitivity. Against the study's other quantities:

| quantity | CER |
|---|---|
| within-run late-training sd (the noise floor used throughout) | 1.06 |
| region-match effect being measured | -0.61 |
| design minimum detectable effect | 4.89 |
| **observed run-to-run spread (n = 1 pair)** | **5.01** |

**`late_sd` measures optimisation noise along one trajectory and systematically understates
between-run variance.** Every claim in this repo that leans on it -- including the statement
that MDE has "converged on the noise floor" -- is optimistic by roughly a factor of five. This
is now the study's most important open uncertainty, and it rests on a single pair.

### Plan

Re-running `am_et` (4) and `crs_sc` (6) yields **11 replicate pairs** with the existing
`en_us`/water one, which is enough to estimate between-run variance directly rather than by
proxy. Two decisions recorded:

- **Vary the seed.** A same-seed re-run measures only nondeterminism; a different seed captures
  nondeterminism plus seed sensitivity, which is the larger and more defensible number.
- **Re-serial only when the replacements land.** Moving `am_et`/`crs_sc` to serial 1 before the
  re-runs finish leaves serial 0 with holes and breaks the current analysis.
  `rename_wandb_serial.py` does this by run id with a dry run.

Serial 1 then means "the earlier run of a cell that has been run more than once", which
`en_us`/water already satisfies, so replicate pairing is a join on (model_id, dataset).

### Caution

`am_et`'s **+0.64** delta is load-bearing: it is the near-controlled comparison against Tamil
(8,873 vs 8,846 clips, opposite sign) that killed the volume hypothesis. With a plausible
5 CER of run-to-run spread, the replicate may well move it. That is what replication is for,
but the conclusion should be held loosely until it lands.

---

## 2026-08-02 (evening) — first baseline numbers, and they are lopsided

Serial 10 is 14 rows in, and the sweep turned out to cover **both** evaluation domains, not just
FLEURS: 11 languages on `google/fleurs` test (cross-domain) and 3 on `disco-eth/WorldSpeech`
test (in-domain, the training domain). The first version of `analyze_baselines.py` collapsed
those, reporting two `en_us` rows as a duplicate cell; it now keys coverage on
(language, domain, model) and normalises WorldSpeech's training-variety names (`fr_ca`, `es_mx`)
onto the study cell (`fr_fr`, `es_419`) via a new `utils.to_study_cell`, without which those
rows lost their resource tier and dropped out of every grouped table.

**Only whisper-medium has run.** Voxtral-Mini and Qwen2-Audio have not started, so nothing here
is a statement about LALM baselines in general -- it is a statement about one ASR model.

### whisper-medium falls off a cliff on the low-resource languages

| language | tier | WER (FLEURS test) |
|---|---|---|
| es_419 | high | 3.55 |
| en_us | high | 5.29 |
| fr_fr | high | 7.91 |
| id_id | high | 10.77 |
| ur_pk | low | 29.74 |
| hi_in | high | 30.19 |
| ta_in | very_low | 31.08 |
| mr_in | mid | 76.48 |
| sw_ke | high | 93.85 |
| ha_ng | mid | 121.97 |
| am_et | very_low | **365.02** |

Amharic at `wip` 0.00 and `mer` 99.99 is not a bad transcript, it is no transcript -- the model
emits text unrelated to the audio. Hausa and Swahili above 90 WER are equally unusable.

That is the strongest argument the project has produced for connector training so far: there is
enormous headroom on exactly the languages the study exists to serve. It is **not yet a result**
-- serial 11 has not run, so the like-for-like contrast does not exist.

### The in-domain corpus is the harder one

For the two languages evaluated on both domains, WorldSpeech test is **2.40x** (en_us) and
**1.89x** (fr_fr) worse than FLEURS test. So the in-domain evaluation is not the easy one, and
adding it will move every number in the unfavourable direction.

### Still missing
- **CER.** The FLEURS configs carry `eval_metrics: ['wer_all']`, so the baselines log WER only
  while the training runs report CER. Adding `'cer'` to that list costs nothing at inference.
- Voxtral-Mini and Qwen2-Audio.
- 9 of 11 in-domain cells.
- Serial 11 entirely.

---

## 2026-08-02 (later) — serial 10 baselines wired in

Eval-only baseline runs started on **serial 10**: whisper-medium, Voxtral-Mini and Qwen2-Audio
evaluated directly on `google/fleurs` **test**, with no connector trained. These answer the
prior question serial 0 structurally cannot -- whether SLAM-style connector training is worth
doing at all, against a model you can download today.

This is exactly the run shape the `wer` / `cer` / `rtfx` / `no_params` column list was kept
for. Serial 0's training runs leave all of them empty; serial 10 fills them and leaves the
HF-style `eval/cer` keys empty instead.

### What the finished run's schema showed
`wer_all` emits `wer`, `mer`, `wil`, `wip` (5.29 / 5.25 / 8.69 / 91.31 for whisper-medium on
`en_us`), alongside `rtfx`, `no_params`, `num_samples`, the `audio_length_s_*` statistics and
the size fields (`bpw`, `total_MB`, `max_memory`). `_step` is 0, so there is no history worth
pulling -- summary only.

### Three serials, three measurements -- and only one valid contrast
- **serial 0** eval/cer on FLEURS *validation*. This is the model-SELECTION curve (best
  checkpoint was chosen over it), so it is neither held-out nor the same metric.
- **serial 10** wer on FLEURS *test*.
- **serial 421** the trained checkpoints over the **same FLEURS test configs**
  (`eval_lalm_decoder_txf.sh`).
  *(CORRECTED same day: the trained sweep uses **serial 11**, not 421 -- 421 was my choice from
  QuantizedASR's 3xx/4xx block, but these are project serials for the LALM study and 11 pairs
  with the 10 baselines. Everything below reads 421; substitute 11.)*

So **421 minus 10** is the like-for-like "what did training buy" contrast, and
`analyze_baselines.py` computes it the moment 421 exists. Comparing serial 10's WER against
serial 0's CER would mix two metrics on two splits and is deliberately refused.

### Added
- **`analyze_baselines.py`** (t7). Handles partial data by construction -- serial 10 fills in
  incrementally, so every cell is reported present/absent rather than assumed, and the contrast
  defers cleanly until serial 421 lands. Compares the *best* baseline per cell rather than the
  mean, since the honest comparator is the strongest downloadable model.
- `plotter.sh` downloads serial 10 **unguarded** (a cached copy goes stale while the sweep is
  still filling in) and tolerates its absence.

### Flagged
The FLEURS configs carry `eval_metrics: ['wer_all']`, so **no CER is logged for the baselines**.
Adding `'cer'` to that list would let the baselines meet the training runs' metric at no extra
inference cost. Worth doing before the sweep completes.

---

## 2026-08-02 — Urdu lands, and the volume hypothesis dies

All 52 runs are finished; the four `ur_pk` runs reached step 1000 with the full 101 evals.

### The headline: the volume interaction did not survive replication

| languages | Spearman rho | p |
|---|---|---|
| 7 | 0.964 | 0.0005 |
| 10 | 0.721 | 0.0186 |
| **11 (+ ur_pk)** | **0.555** | **0.0767** |

Every addition shrank it; it is now not significant even with every language, and falls to
rho 0.406 (p 0.244) without the extreme point. The reason is visible per language: the three
lowest-resource cells are `ta_in` (-14.70), `am_et` (+0.64) and `ur_pk` (+1.44), so **two of
three favour the MISMATCHED decoder**. `am_et` at 8,873 clips against `ta_in`'s 8,846 is
effectively a controlled comparison with the opposite sign. Tamil is an outlier; the trend was
one language plus a seven-point sample.

Region matching itself is a flat null: mean -0.61 CER, Wilcoxon p = **1.000**, MDE 4.89.

### The contrast that is the real result

Data volume **does** predict overfitting, and Urdu strengthened that. With the low tier filled,
median eval-loss rise is monotone across all four tiers -- very_low 0.195, low 0.188, mid 0.028,
high 0.003 -- as is time-to-best-eval-loss (0.275, 0.289, 0.629, 0.815). Meanwhile it does not
predict whether a region-matched decoder helps. Volume governs how badly a cell overfits, not
which decoder to pair with it.

### Added

- **`for_quantizedasr/tools/preprocess/create_yamls_models_lalm_txf.py`** -- crawls the ERISLab
  HF org and generates `configs/models/*_txf_*.yaml` for the trained connector checkpoints,
  following the existing `txf` convention (commented original `model_id`, then `model_id` +
  `local_model` pointing at the Hub repo, `max_new_tokens`/`max_input_length`, `model_dtype`).
  86 checkpoints found: 5 variants x 11 training languages x best-step and step-1000. 82 configs
  written; the 4 `es_es` ones are skipped as superseded (and have no step-1000). Reports that
  `am_et` and `crs_sc` have finished runs but **no uploaded checkpoints**.
- **`for_quantizedasr/scripts/eval_lalm_decoder_txf.sh`** -- serial 421, 156 evaluations. Pairs
  each checkpoint with its OWN training language's FLEURS test and WorldSpeech test rather than
  a full cross product, which would be ~1800 mostly meaningless runs. The pairing is written
  explicitly because `fr_ca -> fleurs_fr_fr` and `es_mx -> fleurs_es_419` are not derivable from
  the config name.

---

## 2026-08-01 — two new languages, a re-serial, and the volume hypothesis weakens

### wandb hygiene
`en_us`/water had been run twice under serial 0, breaking the one-row-per-cell contract. New
**`rename_wandb_serial.py`** (selection by explicit run id, because a re-run is
config-identical to its original) moved the superseded 2026-07-27 run to **serial 1** and
rewrote its name suffix. Dry-run by default; applied and verified.

### The grid is now 12 languages
`am_et` (Amharic, ~40 h) and `ur_pk` (Urdu, ~80 h) fill the low and middle resource tiers, and
`ta_in` gained `base` + `Qwen3-4B`. `RESOURCE_TIER` encodes the four tiers; all four are now
populated, which was the point of the additions.

**Spanish changed underneath the analysis.** The Spain-Spanish (`es_es`) runs were deleted and
re-run from `es_mx`. That halves the cell's stream (866k -> 206k clips) and removes its
dialect-mismatch flag, since Mexican Spanish is inside the Latin American variety FLEURS
`es_419` evaluates.

**`en_us`/water is no longer excluded.** It replicated (12.05 vs 17.06, both far worse than the
other variants), so it is an effect rather than a failed run -- and since water is English's
matched variant, excluding it had been removing the strongest against-hypothesis point.
`EXCLUDED_MODELS_AGGREGATE` is now empty.

### The headline result weakened, and that is the finding
With 7 languages the volume trend was rho 0.96, p 0.0005, robust to dropping Tamil. With 10 it
is rho 0.721, p 0.0186 -- and **no longer robust** (p 0.0769 without Tamil). The cause is one
cell: `am_et` has 8,873 clips against `ta_in`'s 8,846 but a delta of +0.64 versus -14.70. Same
volume, opposite sign. Tamil's outlier is about Tamil, not about being low-resource. Region
matching itself is a clean null: mean -0.81 CER, p 0.695, MDE 5.42.

### Added
- **`analyze_loss_metrics.py`** (t6): separates overfitting (eval loss rising after its own
  minimum) from domain/accent shift (a large train-eval gap with no rise). Overfitting is
  monotone in resource tier (0.195 / 0.028 / 0.003), and cross-domain evals show a 6x larger
  generalisation gap with the same near-zero rise -- the two are separable.
- **`fetch_tinyaya_composition.py`**: parses the Tiny Aya report's Appendix A tables
  (arXiv:2603.11510, Tables 8-14) into a CSV, turning "specialisation" from a categorical label
  into a continuous per-language exposure variable. Stdlib-only parser (no lxml). Validated by
  reproducing each region's printed Subtotal and by checking the mix->variant mapping against
  the numbers. Caught its own bug: `Subtotal` rows were first counted as languages, inflating
  every mix total to ~180 %.
- **`for_quantizedasr/`**: a WorldSpeech in-domain eval config generator and a both-domains
  eval sweep (serial 420), written in QuantizedASR's style but kept here, since that repo is
  not modified from this session.

### Corrections
- The regional definition was wrong: Earth is Africa **+ West Asia**, Water is Asia-Pacific
  **+ Europe**. No executed language is West Asian so no assignment changed, but the definition
  is now recorded correctly.
- Urdu is ambiguous: the report's Table 1 places it in South Asia, Appendix Table 10 under West
  Asia. Assigned `fire` because the fire mix carries 3.4 % Urdu against 1.3 % earth.

---

## 2026-07-30 (fifth pass) — a filter bug becomes the study's central finding

The strict `< 30 s` duration cap silently discarded 72.4 % of the Tamil training data, and
understanding that changed the paper's claim rather than just one table.

### Root cause

`make_audio_length_filter_fn` keeps a clip when `length < max_input_length` — strict — and every
`configs/train/*ws*.yaml` sets `max_input_length: 30`. WorldSpeech `ta_lk` is pre-segmented into
fixed 30-second windows (100/100 sampled rows at exactly 30.00 s), so every clip fails `30.0 < 30`.
Filtering the interleaved Tamil stream leaves exactly **8,846** rows == `len(ta_in)`, out of an
intended 32,107. **23,261 clips lost, with nothing in the logs reporting it.**

This vindicates `analyze_data_accounting.py`, which inferred 8,825 against a true 8,846 (99.76 %).
Against the post-filter expectation all nine languages now reconcile. Written up for upstream in
the new **`docs/UPSTREAM_FIXES.md`**; QuantizedASR is not modified.

### The mistake I nearly shipped

I planned to **exclude** Tamil as a contaminated cell. That was wrong, and the correction is the
finding. The region-match contrast is computed *within* a language: all four variants consumed the
identical 8,846-clip stream, so the loss reduced every arm equally and cannot bias the comparison.
It only relocates Tamil on the data-volume axis, where it is the grid's only genuinely
low-resource cell — the most informative point, not a corrupt one. `utils.py` now carries an
explicit note against re-introducing the exclusion, and a test asserts `ta_in` is in no exclusion
list.

### The finding

Ordering the seven usable languages by training-stream size, the matched-decoder benefit decays
monotonically from **-14.70 CER at 8,846 utterances to +1.06 at 577,382**: Spearman
**rho = 0.964, p = 0.0005**, surviving removal of the extreme point (**rho = 0.943, p = 0.0048**).

Reported with the confound rather than around it: volume and baseline CER are collinear
(rho = -0.679), the *relative* effect trends only weakly (rho = 0.714, p = 0.0713), and the
partial correlation controlling for difficulty is inconclusive over all seven (r = 0.137, p = 0.77,
df = 4) though significant over the six non-extreme languages (r = 0.838, p = 0.04). The decisive
test is a within-language volume manipulation — re-run Tamil at full volume, keep the low-volume
runs — which is exactly the 4-run budget available.

### Added

- **`analyze_volume_interaction.py`** → `t5_volume_interaction.csv`, `t5_volume_stats.csv`.
- **Two figures**: the decay curve and its relative-effect companion, so the confound is visible.
- **`utils.py`**: `MAX_INPUT_LENGTH_S`, `CONFIG_DURATION_AT_CAP` (frozen at-cap screen),
  `KNOWN_AT_CAP_CONFIGS`, and `expected_stream_examples(post_filter=)`.
- **`verify_dataset_durations.py`**: an at-cap screen for any config. Snapshot-first with the live
  datasets-server sample as cross-check, because the endpoint 500s for uncached configs and a
  screen that degrades to "inconclusive, exit 0" is how this bug survived. Known-bad configs are
  reported as `[KNOWN]` so the guard gates on new regressions instead of failing forever.
- 77 unit checks (up from 79 total across both test scripts), 29 ordering claims, 94 numbers.

### Two bugs found in my own code along the way

- **`plot_curve.get_hue_order` was non-deterministic.** It iterated a `set` for hue levels absent
  from `METHODS_DIC`, and CPython randomises string hashing per process — so region colours
  changed between renders while point positions and labels stayed pixel-identical. That is the
  exact silent-colour-shift defect the repo conventions warn about, reproduced by me. Now sorted;
  verified byte-identical across processes, and confirmed the old path really did vary
  (`PYTHONHASHSEED=3` gives a different set order than 1 or 2).
- **A `--load` path that had never run.** `make_audio_length_filter_fn` copied the upstream dict
  access, but under `datasets` 4.x with torchcodec the `Audio` feature yields an `AudioDecoder`,
  so the bare `except` would have marked every clip corrupt. Fixed and exercised for real.

---

## 2026-07-30 (fourth pass) — the WorldSpeech "malformed files" problem is an env problem

Comparing `pip list` from the remote host that *can* read WorldSpeech against the two local
conda envs identified the cause, and it is not libsndfile or Opus.

| | `pytorch` (local) | `asr` (local) | remote host |
|---|---|---|---|
| `datasets` | 5.0.0 | 4.5.0 | 4.5.0 |
| `transformers` | 5.14.1 | 4.57.5 | 4.57.5 |
| `soundfile` | **absent** | 0.14.0 | 0.13.1 |
| `torchcodec` | **absent** | 0.9.1 | 0.9.1 |
| `torchaudio` | **absent** | absent | 2.9.1+cu130 |
| ffmpeg/libsndfile `.so` | **0** | 18 | — |
| `scipy` | present | **absent** | — |

`pytorch` has **no audio backend whatsoever**: `datasets` 5.0.0 decodes `Audio` through
torchcodec, and neither torchcodec nor soundfile is installed, so *every* audio read fails
there regardless of codec. `asr` matches the remote host closely — including `transformers`
4.57.5, which is exactly the `transformers_version` logged by the runs — so `asr` is the env
that reproduces how training consumed data.

**Retested in `asr`, and the blocker is gone.** The three configs recorded upstream as
undecodable (`examples/explore_datasets.py:30`, "Supported file format but file is malformed")
all decode correctly via torchcodec at 24 kHz: `la_va` 29.54 s, `si_lk` 29.34 s, `tl_ph` 0.80 s,
and `ta_in` 18.01 s for the first test clip of each. The libsndfile error string means an env
without torchcodec fell back to soundfile; it was never a property of the corpus.

Consequences:

- **`CLAUDE.md` corrected.** It stated "There is no `asr` env on this machine, whatever older
  docs say." Both halves were wrong: `asr` exists and it is the only local env that can read
  audio. The file now carries the split — analysis and `plotter.sh` in `pytorch` (it has scipy,
  which `asr` lacks), anything loading a dataset in `asr`.
- **`EVAL_DATASET_PLAN.md`'s top-priority tier is unblocked.** Adding matched WorldSpeech `test`
  eval configs was written up as blocked on a decode bug; it is ordinary work.
- **The interleave guard is version-robust.** It was originally verified on `datasets` 5.0.0;
  re-run under `asr`/`datasets` 4.5.0 — the version the runs actually used — it passes
  identically, which matters because the earlier verification and the runs were on different
  major versions.
- **A real bug in `verify_dataset_durations.py`, found because the env difference forced the
  question.** Its `make_audio_length_fn` was a verbatim copy of the upstream function, which
  reads `audio['array']`. Under `datasets` 4.x with torchcodec the `Audio` feature yields an
  `AudioDecoder` instead, so dict access raises and the bare `except` would have assigned the
  corrupt sentinel to **every** clip — reporting a whole corpus as broken and filtering the
  split away. The `--load` path had never been exercised, so this was live. It now handles both
  shapes, and `--load` has been run for real: `ta_in` + `ta_lk` `test`, 1690 == 466 + 1224,
  0 undecodable, 0 removed by the duration filter, loaded counts matching builder metadata.
  That independently reproduces on `test` what was reported on `train`.

---

## 2026-07-30 (third pass) — the ta_in "anomaly" was mine, not the data's

A direct test on the real datasets refuted the last remaining suspicion about `ta_in`, and the
input that produced it has been replaced.

### What was tested, and what it showed

`disco-eth/WorldSpeech`, configs `ta_in` + `ta_lk`, split `train`, loaded map-style with
`audio_length_s` computed from the decoded arrays:

* `len(interleaved) == len(ta_in) + len(ta_lk)` — interleaving loses nothing on real data.
* The duration-consistency filter (`|audio_length_s − duration| < 1 s`) removed **zero**
  samples — no corrupt clips, no mislabelled durations.

Both mechanisms the previous pass had proposed for a Tamil data problem are therefore false.

### The bad input: hours recovered from prose

The flag came from comparing a reconstructed stream size against per-language hour figures
taken from a **summarised web fetch** of the WorldSpeech paper's table — not from reading the
table. That was the weakest-provenance input in the whole analysis, and it was gating a
finding. It claimed 240 h for `ta_lk`, which cannot be right for 23,261 clips.

Replaced by `utils.WORLDSPEECH_TRAIN_EXAMPLES`: `num_examples` per training config, read from
the HuggingFace dataset builder metadata. Authoritative, instant, no audio downloaded. Also
added `TRAIN_CONFIGS` (the `(path, configs, split)` each language trained on) and
`expected_stream_examples()`, which sums the parts — valid precisely because interleaving is
lossless.

### What the corrected accounting says

Eight of nine languages reconcile, including the other two multi-config languages, whose
reconstructions match their **summed** streams (`ha_ng` 1.04, `sw_ke` 1.20) and not a single
config (1.84, 1.81). `ta_in` is the exception, and its reconstruction matches **one config to
within 0.24 %** (ratio to expected 0.27; to nearest single config 1.00).

That is a bookkeeping question, not a data question: either the run consumed one config, or the
epoch counter is unreliable for it, and the logs do not separate the two. The only consequence
kept in the analysis is that Tamil is the smallest-data cell, has the worst CER, and carries
the largest region-match term (−14.70) — so that term is provisional on **data-volume**
grounds, not data-quality grounds. `t4`'s docstring now states explicitly that it is not an
integrity check.

### New: `verify_dataset_durations.py`

The test above, generalised into a checker for any `--dataset_path --dataset_configs --split`,
so the next suspicion of this kind is tested before it is written down. Two modes: metadata
only (instant, authoritative counts, wired into `plotter.sh`) and `--load` (downloads audio,
asserts the interleave arithmetic on real objects, and reports exactly how many samples the
duration filter removes). The `--load` invocation is recorded in `claude_process.sh`.

### Also caught

`analyze_region_match.py` still referenced a renamed column and `set -e` — added in the
previous pass — aborted the run rather than letting it through. The checker then flagged a
`crs_sc` ratio that is legitimately undefined (its ERISLab mirror split is not in the snapshot)
and is now excluded from that specific check rather than being papered over.

Rebuild after the corrections: exit 0, **79** checks pass, 21 ordering claims unbroken, 85
numbers verified against their CSVs.

---

## 2026-07-30 (later) — three corrections after review

Three claims from the first pass were wrong or misplaced. All are corrected in code, not just
in prose, so they cannot quietly return.

### 1. Interleaving is NOT a confound — refuted empirically

The first pass flagged `ta_in`/`ha_ng`/`sw_ke` as `uniform_interleave`, reasoning from the
loader that uniform `1/N` probabilities would oversample the smaller config. That reading was
wrong. The strategy is `all_exhausted_without_replacement`, a real supported strategy
(`datasets` 5.0.0 lists it in the `interleave_datasets` signature), and it never recycles an
exhausted dataset — so the combined stream is exactly the sum of its parts.

**`verify_interleave_semantics.py`** (new, offline, in `plotter.sh`) proves it with exact
counts: interleaving a 100-example and a 25-example dataset yields exactly 125 distinct
examples, every example once, in both the map-style and streaming paths, and exact for a
three-way pairing. Plain `all_exhausted` with the same uniform probabilities oversamples the
smaller dataset to 122× (total 222) — the behaviour originally assumed. Verified against the
loader at commit `ac7566e`, which is what the live experiments run.

`TRAIN_EVAL_MATCH` no longer carries the flag, `MULTI_CONFIG_TRAIN` records the pairings
without implying oversampling, and a test asserts the flag stays absent.

### 2. The `crs_sc` `_clean` splits are all genuinely cleaned

The first pass claimed `val_clean` was unfiltered because the committed upstream example only
demonstrates the duration-consistency filter on the `test` split. The filter was in fact
applied to train and val too when the splits were built. `analyze_ood_crs.py` now records this
as a note rather than a caveat, and the docs are corrected.

### 3. Hours/epochs analysis moved here, and no longer needs the training framework

The earlier plan proposed a dataset-hours utility inside QuantizedASR. That repo is not to be
touched. **`analyze_data_accounting.py`** (new) reconstructs each training stream from logged
scalars alone — `global_step × batch_size × gradient_accumulation_steps`, audio seconds, and
the epoch counter, which under streaming counts stream consumptions — and compares it against
a **frozen WorldSpeech hour snapshot** in `utils.WORLDSPEECH_HOURS` (arXiv:2605.09167,
*since removed -- replaced by `WORLDSPEECH_TRAIN_EXAMPLES`, see the third-pass entry*;
copied as objective raw data, never referenced live).

Each entry carries a `scope`, which is load-bearing: only four languages support a direct
config-to-config comparison, three are lower bounds (a second config's hours are unknown), and
three (`en_us`, `fr_fr`, `es_419`) have only a language-level aggregate over configs the runs
never used. Comparing without reading the scope manufactures discrepancies.

Result: **`ta_in` is the sole anomaly** — implied stream 34.82 h against a lower bound of
240 h, ratio 0.15, so the true shortfall exceeds 6.7×.
*(SUPERSEDED by the third pass: the 240 h figure came from a summarised reading of the paper
and was wrong. Direct testing shows the Tamil data is clean. See the entry above.)* Every other comparable language
reconciles (`ha_ng` 1.00, `crs_sc` 1.03, `sw_ke` 1.04, `id_id` 1.28; `mr_in` 2.07 is above
published, which is not a data-loss failure). With interleaving ruled out, the remaining
candidate is the silent filter path: clips of 30 s or more dropped, and undecodable clips
removed via the 100,000 s sentinel without being counted. `ta_in` received no cleaning pass.
This matters because `ta_in` carries the largest region-match term (−14.70 CER).

### 4. `plotter.sh` was not actually gating on its guards

Found while fixing the above: the script had `set -u` but not `set -e`, so its exit status was
only the last command's. A failing `test_utils_port.py` printed `[FAIL]`, exited 1, and
`plotter.sh` still reported success. Now `set -eu`, and verified by deliberately breaking a
check: the run aborts with exit 1 before reaching `verify_paper_numbers.py`.

### Verification after the corrections

`bash plotter.sh` from an emptied `results_all/`: exit 0, **77** unit/semantic checks pass with
0 failures, 19 ordering claims unbroken, 75 numbers verified against their CSVs, 4 figures
regenerated. The checker again caught a drifted number while these edits were made (`id_id`
epochs 5.015 printed as 5.01; correct is 5.02) and a stale ordering claim that assumed
`es_419` appears in the by-language table when it has no finished runs.

---

## 2026-07-30 — first data pipeline

The repo went from seeded-but-empty to a working pipeline over wandb `LisTAya/LALMDecoder`
serial 0 (41 runs). `bash plotter.sh` now rebuilds everything from a bare checkout.

### Added

- **`download_wandb_history.py`** — the time-series download, gated behind `--history`. One
  row per (run, step) via `run.scan_history()`. This is the first history-aware tooling in
  the `/home/edwinrios/analysis/` family; nothing else touches wandb history. ~68 s and
  ~7.5k rows for serial 0.
- **`plot_curve.py`** — training-curve plotting. Separate from `plot.py` because that
  script's `line` branch force-melts onto a hardcoded metric list, `keep_columns` drops the
  needed columns, and `rename_vars` rewrites the axis arguments; editing it would change
  every existing sibling-repo figure. Reuses the same theme, style arguments and save
  convention. Adds faceting, an **explicit** `--errorbar` (seaborn's silent default is
  `('ci', 95)`), the positive-value log-axis fix, and `mkdir -p` on the output subdirectory.
- **`analyze_sample_efficiency.py`** → `t1_sample_efficiency.csv`
- **`analyze_region_match.py`** → `t2_region_match.csv`, `t2_region_match_stats.csv`
- **`analyze_ood_crs.py`** → `t3_crs_ood.csv`
- **`test_utils_port.py`** (52 checks) and **`verify_paper_numbers.py`** (12 ordering claims,
  46 numbers), both wired into `plotter.sh`. Built alongside the first table rather than
  retrofitted.
- **`plotter.sh`**, **`claude_process.sh`**, `NUMBER_PROVENANCE.md`,
  `REPRODUCE_FIGURES.md`, `docs/PLAN_ASSESSMENT.md`, `docs/EVAL_DATASET_PLAN.md`.

### Changed

- **`utils.py`** — placeholder dicts populated from the real runs. The seeded `METHODS_DIC`
  keys (`q2a_whisper_small_tiny_aya_*`) matched **nothing**: the runs use whisper-**medium**
  and full HF paths. Added `MODEL_SHORT`, `CORE_VARIANTS`, `LANGUAGE_REGION`,
  `LANGUAGE_STATUS`, `LANGUAGE_DIC`, `TRAIN_EVAL_MATCH`, and the helpers `half_up`,
  `assert_unique_keys`, `add_language_columns`. `EXCLUDED_MODELS_AGGREGATE` now records the
  failed `en_us`/`water` run with its reason.
- No changes to `download_save_wandb_data.py`, `concat_df.py`, `count_data.py`, `plot.py` —
  their defaults are all CLI-overridable, so `plotter.sh` passes the right key set per run
  type.

### Findings worth not rediscovering

- **The seeded `SUMMARY_COLS` return nothing for training runs.** Real keys are `eval/cer`,
  `eval/loss`, `train/train_audio_seconds`; `no_params` is `model/num_parameters` and lives
  in *config*. The `wer`/`cer`/`rtfx`/`no_params` columns are kept because planned eval-only
  runs will populate exactly those.
- **Effective batch is 512** (`batch_size` 8 × `gradient_accumulation_steps` 64). Missing the
  accumulation factor makes seconds-of-audio-per-sample come out at ~956 s against a 30 s
  cap — impossible, and the sign that the factor was dropped. `test_utils_port.py` pins it.
- **Unique hours and epochs are not derivable from wandb here.** `train/train_audio_seconds`
  is a real measurement, but `train/epoch` is an estimate under `streaming=True`.
  *(Partly superseded by the later entry: `t4_data_accounting` does reconstruct stream hours
  from the epoch counter, but always tagged `estimate` or `lower_bound`, and compared only
  against published figures whose `scope` permits it. Curves still use audio processed.)*
- **The endpoint comparison is underpowered.** Median within-run late-training CER sd is 1.03
  against a mean region effect of 0.48; minimum detectable effect is 6.97–9.27 CER.
- **Confounds encoded in `TRAIN_EVAL_MATCH`**: `fr_fr`/`es_419` are cross-dialect
  (trained `fr_ca`/`es_es`, evaluated `fr_fr`/`es_419`).
  *(SUPERSEDED in part: this entry also claimed `ta_in`/`ha_ng`/`sw_ke` were confounded by
  50/50 interleaving. That was wrong — see the later entry and
  `verify_interleave_semantics.py`. The flag has been removed.)*
- ~~**`crs_sc`'s `val_clean` split is not duration-filtered** despite the name.~~
  *(WRONG, corrected in the later entry: train, val and test all carry the cleaning.)*

### Verification

- `bash plotter.sh` from an emptied `results_all/`: exit 0, 52 unit checks pass, 12 ordering
  claims unbroken, 46 numbers verified, 4 figures regenerated.
- Both checkers negative-tested (`claude_process.sh`, spent lines commented out). The first
  attempt perturbed t1 row 0 and the checker stayed green — correctly, since row 0 is
  `crs_sc`/`base` and every checked aggregate is over `CORE_VARIANTS`. Retargeted; all three
  perturbations then went red and restored green.
- The checker caught three drifted numbers in `PLAN_ASSESSMENT.md` on first run, including a
  double-rounding defect (`0.644882` written as `0.65`; correct is `0.64`) — the exact class
  of bug the conventions warn about.
