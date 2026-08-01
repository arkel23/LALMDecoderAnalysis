# Fixes needed in QuantizedASR

Written 2026-07-30 from this repo's analysis. **QuantizedASR is deliberately not modified here** —
this file records what to change there, why, and how to verify it.

---

## 1. The duration cap deletes clips that sit exactly on it (data loss, high priority)

### What happens

`qasr/data/data_utils.py:84-94`:

```python
def make_audio_length_filter_fn(min_input_length=1, max_input_length=30):
    def filter_audio(length):
        if min_input_length and max_input_length:
            in_range = length > min_input_length and length < max_input_length
        elif min_input_length:
            in_range = length > min_input_length
        elif max_input_length:
            in_range = length < max_input_length      # <-- strict
        return in_range
    return filter_audio
```

The upper bound is **strict**, so a clip of exactly `max_input_length` seconds is discarded. All
ten `configs/train/*ws*.yaml` set `max_input_length: 30`.

WorldSpeech `ta_lk` is pre-segmented into fixed 30-second windows — 100/100 sampled rows are at
exactly 30.00 s. So the filter removes **every** `ta_lk` clip.

### Measured impact

The Tamil training config interleaves `ta_in` + `ta_lk`. Filtering the interleaved stream with
the real training filter leaves exactly **8,846** rows, i.e. `len(ta_in)`:

| | clips |
|---|---|
| `ta_in` train | 8,846 |
| `ta_lk` train | 23,261 |
| interleaved (verified lossless) | 32,107 |
| **after `max_input_length: 30`** | **8,846** |

**23,261 clips — 72.4 % of the intended Tamil training data — were silently discarded.** Nothing
in the logs reports it: the drop happens inside a `datasets.filter`, and undecodable clips are
removed by the same filter via a `1e5` sentinel, so the count is never surfaced.

Downstream consequence in this study: Tamil became the only genuinely low-resource cell, produced
the worst CER in the grid, and generated a −14.70 CER region-match difference that was on course
to be reported as the headline result.

### Blast radius across the configs in use

Screened without downloading audio (datasets-server `rows`/`statistics`); frozen in
`utils.CONFIG_DURATION_AT_CAP`.

| config | clips at/above 30 s | note |
|---|---|---|
| `ta_lk` | **100 %** | every clip exactly 30.00 s — total loss |
| `fr_ca` | ~4 % | 4/100 sampled, max 52.54 s |
| `ta_in` | 0 % | proven: all 8,846 survive the filter |
| `hi_in`, `sw_ke`, `ha_ng` | 0 % | 0/100 sampled at the cap |
| `en_us`, `es_es`, `es_mx`, `sw_tz` | 0 % | max duration below the cap (20.00 / 25.00 / 20.00 / 29.98 s) |
| `ha_td`, `mr_in`, `id_id` | not sampled | endpoint uncached; each language's stream reconciles with its full example count, which is independent evidence the cap is not removing a material share |

### The fix

Make the upper bound inclusive, which is almost certainly the intent — `max_input_length: 30`
reads as "clips up to 30 seconds", not "clips under 30 seconds":

```python
            in_range = length > min_input_length and length <= max_input_length
...
            in_range = length <= max_input_length
```

The config-level alternative (`max_input_length: 30.001` in the ten `configs/train/*ws*.yaml`)
avoids touching shared code but leaves the trap in place for the next fixed-window corpus.
Prefer the code fix.

Note that Whisper pads to 30 s regardless, so admitting exactly-30 s clips changes nothing
about the model's input shape — only how many examples reach it.

### Regression test worth adding

```python
f = make_audio_length_filter_fn(min_input_length=None, max_input_length=30)
assert f(29.99)          # under the cap
assert f(30.0)           # AT the cap -- this is the bug; fails before the fix
assert not f(30.01)      # over the cap
```

Also worth guarding: with both bounds falsy, `filter_audio` raises `UnboundLocalError` because
`in_range` is never assigned. `prepare_data` currently avoids this only by wrapping the call in
`if min_input_length or max_input_length`.

### Prevention

`verify_dataset_durations.py` in this repo screens any `--dataset_path --dataset_configs
--split` for the fraction of clips at/above the cap and fails when it exceeds a threshold. It
reads a frozen snapshot first and uses the live datasets-server sample only as a cross-check,
so it stays deterministic and offline. Worth mirroring upstream, or at least running before
adding a new training config.

---

## 2. Stale documentation: WorldSpeech is not undecodable

`README.md:356-357` says the three `create_yamls_worldspeech_gaps` configs are unusable:

> every example hit a `libsndfile`/Opus decode error in this environment (`libsndfile 1.0.31`
> lacks solid Opus support - an environment limitation, not a dataset problem)

and `examples/explore_datasets.py:30` carries the same note.

Both are stale on two counts. The pinned `soundfile==0.13.1` wheel bundles **libsndfile 1.2.2**,
not 1.0.31, and 1.2.2 supports Opus. More fundamentally, with `torchcodec==0.9.1` installed,
`datasets` 4.5.0 decodes `Audio` through torchcodec/FFmpeg and never touches libsndfile.

Retested 2026-07-30 in an env matching `requirements.txt`: all three decode correctly at 24 kHz —
`la_va` 29.54 s, `si_lk` 29.34 s, `tl_ph` 0.80 s on the first test clip of each.

So `worldspeech_{la_va,si_lk,tl_ph}_test.yaml` can be wired into an `eval_*.sh` script, and the
note should be corrected.

The `libsndfile` error string is what an environment **without** torchcodec produces when
`datasets` falls back to soundfile — i.e. it diagnoses a missing dependency, not a corpus.

---

## 3. Nothing to add to `requirements.txt`

Checked: it already pins `datasets==4.5.0`, `soundfile==0.13.1`, `torchcodec==0.9.1`,
`torchaudio==2.9.1+cu130` and `transformers==4.57.5`, and `README.md:19-22` already documents
the one non-pip prerequisite (`conda install -c conda-forge 'ffmpeg<8'`) with the correct
reasoning — `pip show torchcodec` reports no dependencies, so pip cannot enforce FFmpeg.
`soundfile` bundles its own libsndfile, so it needs nothing extra.
