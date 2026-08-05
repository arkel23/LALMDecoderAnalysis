# LALMDecoderAnalysis — conventions

Analysis repo for the LisTAya / TinyAya decoder-SFT comparison. See `HANDOVER.md` for the
project goals and the tools available here.

These conventions are inherited from `/home/edwinrios/analysis/CLAUDE.md` (the shared file
for the QASR analysis repos), pruned to what applies to a project that has **not produced
data yet**. The parent file still governs anything not covered here; re-read it if this repo
grows a paper.

## Environment

- Work happens **locally**, under `/home/edwinrios/analysis/` — not `/mnt/c/...` (far less
  disk) and not the remote host. Anything CPU-bound (metrics, correlations, paper builds)
  runs here.
- Conda envs: **two, and which one you need depends on the task.** (The previous claim here --
  "there is no `asr` env on this machine" -- was wrong. It exists, and it is the only one that
  can read audio.)

  | | `pytorch` | `asr` |
  |---|---|---|
  | `datasets` | 5.0.0 | **4.5.0** (what the training runs used) |
  | `transformers` | 5.14.1 | **4.57.5** (matches `transformers_version` in wandb) |
  | `soundfile` / `torchcodec` | **absent / absent** | 0.14.0 / **0.9.1** |
  | ffmpeg+libsndfile shared objects | **0** | 18 |
  | `scipy` | present | **absent** |

  - **Analysis, figures, verification -> `pytorch`.** It has scipy, which
    `analyze_region_match.py` needs. `bash plotter.sh` runs here.
  - **Anything that loads a dataset -> `asr`.** `pytorch` has *no audio backend at all*
    (`datasets` 5.0.0 with neither soundfile nor torchcodec), so every audio read fails there.
    `verify_dataset_durations.py --load` must run in `asr`.
  - `asr` closely matches the remote GPU host, including `transformers` 4.57.5, so it is the
    right env for reproducing anything about how the training runs consumed data.
- The remote host `ubuntu@140.114.79.186` (`server-3090`, 1 x RTX 3090 24 GB) is for
  **GPU work and large-scale jobs**. As of 2026-07-31 all downloads, checkpoints and
  experiment scratch go under **`/hdd10tb/edwin/`** (9.1 TB) — one subfolder per job.
  Never `~`, never `/` (~91% full). The old `/hdd/edwin/` and `/media/samsung/` paths
  are superseded; `/media/samsung` is nearly full. Use `conda activate asr` there.
  For big downloads prefer `hf download <repo> --local-dir <dir>`.
- LaTeX: TeX Live 2025, `pdflatex`/`bibtex` work. `pdftoppm` is **not** installed; use
  `mutool draw -r 110 -o out%d.png main.pdf 1-N` to render pages.

## Do

- Create new files for new functionality; prefer additive changes over editing existing ones.
- Follow each repo's existing style.
- Ask when it's unclear whether an existing file should be touched.

## Do not

- Do not delete files unless explicitly told to. If something looks obsolete, move it to
  `deprecated/` with a dated `README.md` entry explaining why and what would restore it.
- Do not add `Co-Authored-By: Claude` or any AI-attribution trailer, or a "Generated with
  Claude Code" footer, to commits or PR descriptions. Commit messages describe the change,
  nothing else.
- Do not assume remote credentials beyond what's given.
- Do not reimplement a statistic ad hoc when a repo function exists (`compute_correlations.py`
  and friends) — a from-scratch reimplementation introduced a fresh bug at least once.

## Git

Commit and push are **allowed** (changed 2026-07-28; the old standing "never commit" rule is
retired). Conventions:

- Push work to a **branch for review**, not straight to `main`, unless told otherwise.
- Author identity is set **globally on this machine**: `Edwin Arkel Rios
  <edwinarkel.rios@gmail.com>`. Use it; no per-repo override is needed. (It was unset before
  2026-07-28, which made the first commit fail with "Author identity unknown".)
- Existing history shows two other authors, both expected and not to be corrected:
  `ubuntu <calvinku1209@gmail.com>` on older commits (the old remote host's default) and
  `Edwin Rios <edwin.rios@tranxform.com>` on MultilingualQASR's two 2026-07-28 commits, made
  before the default above was established. Fixing them would mean rewriting published
  history — don't.
- Never rewrite published history. Both MultilingualQASR and ChineseQASR have the raw
  transcript corpus committed (`.git` is hundreds of MB). It is accepted; don't "fix" it.

## The core principle: duplicate, don't share

This is one-time, per-paper code. Once a paper is submitted its analysis is done.

- **Never** extract shared abstractions across sibling repos, and never import from one into
  another. Starting a new paper means copying a repo wholesale and pruning it.
- The only thing reasonably shared is objective, non-analytical raw data (e.g. Whisper
  training-hours-by-language) — copied as a **frozen snapshot** at fork time, never
  referenced live, so a correction for one paper can't silently reopen a finished one.
- Each repo must be **standalone to a testable bar**: one `bash plotter.sh` from a bare
  checkout rebuilds everything — download, concat, every analysis, every figure, the copy of
  those figures into `paper/`, and the verification guards. A new analysis step must be wired
  in; never leave a "run this by hand first" step.

## Paper-revision checklist

Every item below corresponds to a defect that actually shipped in these repos.

**Provenance — no hand-computed numbers**

1. Every printed number traces to a **script that writes a CSV**. Not an ad-hoc snippet, and
   not a snippet living in a markdown file (one headline figure hid there for weeks).
2. Keep a **machine check** (`verify_paper_numbers.py`) that re-derives every printed number
   from its CSV and fails on drift. Adding a number to the paper means adding it to the spec.
3. The checker must **derive** expected values from the CSV, never hardcode them.
4. Round **half-up**, not Python's banker's default — they differ on ties (5.25).
5. **Store enough precision** that one rounding is correct. 2-dp storage turned a true
   68.2479 into 68.25 into a wrong 68.3, and 3.644982 into 3.645 into a wrong 3.65. Use 6 dp
   for statistics.
6. A number the paper prints must live in a **CSV, not a script's stdout**.
7. Verify **derived** values too — ratios, ranges, subset medians, counts. Nothing regenerates
   them, so they rot silently.
8. Verify **orderings and monotonicity** ("X leads at every tier", "cost grows as bits drop").
   No printed digit changes when an ordering flips, so these are what a data correction
   invalidates without trace.
9. **Negative-test the checker**: perturb a CSV, confirm the run goes red, restore. A checker
   that has never failed is not a checker.
10. Report **low-specificity** checks rather than counting them as wins — a document-wide
    search for "9" or "3.5" proves little. An honest coverage number beats a flattering one.

**Correctness**

11. Every merge or paired subtraction **asserts key uniqueness**. A wrong key silently
    produced k² cross-dataset pairs and corrupted a headline table; the failure is invisible
    in the output, which just looks like a larger, more reassuring *n*.
12. Every **null claim carries a statistical test**. "These medians look similar" is the
    weakest form of the argument and the first thing a reviewer attacks.
13. State the **aggregation** each table uses — per-run, per-language, or utterance-weighted
    micro-average. The same model legitimately scored 8.2, 7.3 and 8.4 under the three.
14. Apply declared filters **consistently** (a figure caption once quoted pre-outlier-filter
    numbers while the paper declared a cutoff).
15. **Unit-test the logic**, not just the outputs. A number-checker cannot catch a CSV that is
    *consistently* wrong — a mis-tiered language regenerates every table wrongly and still
    passes every check.

**Staleness — the most recurrent failure**

16. After **any** upstream data correction, regenerate every derived CSV and re-run the
    checks. Three tables were transcribed from CSVs never regenerated after a ≤0.005pp fix
    that was still enough to flip printed roundings.
17. When a number changes for a surprising reason, **prove the cause** before accepting it.

**Presentation**

18. **Compile *and visually render* changed pages.** LaTeX reported 0 errors while two tables
    physically overlapped, and again while a table bled into the adjacent column.
19. Figures in `paper/` must be **byte-identical** to their generated sources, and copied by
    the pipeline rather than by hand.
20. Report dataset and model statistics tables — expected of any benchmark paper.
21. No self-narration of revision history ("we previously found", "an earlier draft") and no
    implementation identifiers (script names, `\texttt{}`-wrapped code) in prose.

## Less is more (2026-08-03 review — read this before writing any file)

A generator here reached **245 lines doing what QuantizedASR's `create_yamls_short_ml.py` does
in 75**, and seven docs totalled 1,695 lines. The verdict: too much prose, too little signal.
Comments that mix necessary documentation with narrative make it *impossible to tell which is
which*, so none of it gets read.

**Code**
- Match the reference file's size and shape. For config generators that is
  `tools/preprocess/create_yamls_short_ml.py`: explicit parallel lists, one loop, near-zero
  comment. If a script is 3x its reference, it is wrong.
- One script does **one thing**. A config generator generates configs — it does not also emit
  manifests, pairing tables, or study metadata.
- Comment budget: **~2 lines at a site**, and only for a non-obvious invariant. If a reader
  could derive it from the code, delete it.
- No narrative in code. "Why I changed this", "what the old version did", "how the bug was
  found", "what this cost" — all of that goes in `docs/CLAUDE_CHANGES.md`, never in a
  docstring or a comment block.
- Don't regenerate what exists. Check the target repo first (`configs/datasets/short_ml/`
  already holds 229 FLEURS configs); reference existing files instead of re-emitting them.

**Docs**
- Prefer editing an existing doc to adding a new one. Ask which existing file this belongs in
  before creating a file.
- Each doc needs a distinct niche. If two overlap, merge them.
- `docs/CLAUDE_CHANGES.md` is the only place for chronology and rationale.

**Replies**
- Short. No restating a point in a second form, no summarising what was just said.

### A worked example of getting this wrong (2026-08-03, LALMDecoderAnalysis)

A config generator was written at **245 lines**. A reference doing the same job already existed
-- `QuantizedASR/tools/preprocess/create_yamls_short_ml.py` -- at **75**. The extra 170 lines:

- a 40-line module docstring covering where the file lives and why, why the configs were needed,
  and which earlier belief about the corpus had turned out wrong;
- inline notes on list entries that already said the same thing (`('en_au', ..., 'Australian')`);
- two extra jobs bolted on -- emitting a manifest CSV and a pairing table -- that nobody asked a
  config generator to do.

Rewritten to the reference's shape, one explicit list and one loop, it came to **58 lines** and
did strictly more: 120 configs instead of 33. Nothing was lost, because none of the deleted prose
stated an invariant. In the same session a bash sweep carried **60 lines of comment before its
first statement**, and a docstring narrated a correction the user had given ("I previously
claimed X was infeasible; that answered the wrong question").

Two related failures from the same session:

- **Seven documents, 1,695 lines, overlapping.** Two of them disagreed: one still presented a
  hypothesis as the recommended framing that the other recorded as dead. Consolidated to five.
- **Regenerating what already existed.** 229 FLEURS configs were in the target repo; the
  generator emitted its own anyway. Check before generating.

**The test, before writing any comment:**

1. Could the reader derive this from the code? Delete it.
2. Could they only know it from a conversation, a commit, or an incident? It belongs in
   `docs/CLAUDE_CHANGES.md`, not here.
3. What survives is a non-obvious invariant. That is the comment, and it is usually one line.

## Verification discipline

- Independently re-derive any numeric or figure claim before trusting it — re-run the
  generating script fresh and diff against what the paper shows.
- After changing `main.tex`: `pdflatex → bibtex → pdflatex ×2`, then confirm zero `^!` errors,
  zero undefined references, expected page count, **and render the changed pages**.
- After changing `utils.py`/`plot.py`, regenerate affected figures and check legend, axes
  **and colors** — a duplicate-containing `hue_order` once shifted one category's color while
  positions and labels stayed pixel-identical.
- Update `NUMBER_PROVENANCE.md`/`REPRODUCE_FIGURES.md` whenever a number or figure changes.
- Log substantive changes, dated, in `docs/CLAUDE_CHANGES.md`.

## Session workflow

- One-off commands go in the repo's **`claude_process.sh`** with a dated comment, and get
  **commented out once spent** — three live "DONE" lines would have relaunched a multi-hour
  recompute on the next `bash claude_process.sh`.
- Don't wait sequentially on independent multi-step batches. Start a downstream step on
  whatever has finished upstream; use extra scratch `.sh` files if needed and consolidate
  after.
- Long jobs: `nohup ... &` with a log, and print incremental progress so a killed run is
  diagnosable. Check memory first — a 15M-utterance pooling step needs ~4 GB and will be
  OOM-killed on a loaded box.
- Re-running a smoke test uses the **same serials** as the original unless the purpose
  differs, so a human can find the runs where they were said to be.

## When delegating to a background agent or fork

- Give exact methodology (exact filters, groupby keys, serial and column names). Vague
  instructions produced a from-scratch reimplementation with a new bug.
- Independently verify anything numeric it reports, especially aggregations and anything
  contradicting an instruction you gave. Read the deployed file; don't trust the summary.
- Sequence agents that touch the same file; don't edit one `main.tex` from two directions.

## Not yet applicable

Several conventions in the parent file assume a repo with data and a paper. They apply as
soon as those exist, and are listed here so they are not forgotten rather than rediscovered:

- **`plotter.sh`** does not exist yet. Create it as soon as there is a first wandb serial to
  download, and keep the "one command rebuilds everything" bar from day one — it is far
  cheaper than retrofitting it, which is exactly what the sibling repos had to do.
- **`verify_paper_numbers.py`** and unit tests: build them alongside the first table, not
  after. In MultilingualQASR they were retrofitted and immediately found five wrong printed
  numbers.
- **`NUMBER_PROVENANCE.md` / `REPRODUCE_FIGURES.md`**: start them with the first number.
- **`claude_process.sh`**: start it with the first one-off command.

## This repo's specific traps

- **`utils.py`'s dicts are placeholders.** `SERIAL_DIC` is empty and `METHODS_DIC`,
  `DATASETS_DIC` describe the *intended* TinyAya matrix, not observed runs. Fill them from
  real wandb data before trusting any figure. `get_canonical_labels` returns an empty list
  until `SERIAL_DIC` is populated.
- **The multilingual machinery is deliberately gone.** No language-hours table, no resource
  tiers, no `NEEDS_CER` primary-error-rate switch. If this project needs per-language
  scoring, add it deliberately rather than copying MultilingualQASR's version, whose
  `NEEDS_CER` logic never actually fired on its own short-form data.
- **The comparison is only causal if the variants are matched.** See `HANDOVER.md`.
