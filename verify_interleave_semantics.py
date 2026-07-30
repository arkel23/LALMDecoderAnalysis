"""
Empirical verification of what the training runs' dataset interleaving actually does.

Why this exists. The multi-config languages in serial 0 -- ta_in+ta_lk, ha_ng+ha_td,
sw_ke+sw_tz -- are loaded by interleaving two WorldSpeech configs. Reading the upstream
loader, the interleave is called with UNIFORM probabilities (the size-proportional line is
commented out), which looks like it would oversample the smaller config and distort every
per-language number. That reading is wrong, and this script is the proof.

The upstream call, verbatim from qasr/data/data_utils.py at commit ac7566e (the commit the
live experiments run from):

    dataset = interleave_datasets(
        datasets, probabilities=[1/len(dataset_path) for _ in range(len(dataset_path))],
        stopping_strategy='all_exhausted_without_replacement', seed=0
    )

The load-bearing part is 'all_exhausted_without_replacement'. It is a real, supported
stopping strategy (datasets >= 5.0.0 lists it in the interleave_datasets signature), and
"without replacement" means an exhausted dataset is NOT recycled: sampling continues from
whichever datasets still have examples, until all are exhausted. So every example from every
config is yielded exactly once and nothing is duplicated -- the uniform probabilities affect
only the ORDER in which examples arrive, not how many times each is seen.

That is the difference that matters. Under plain 'all_exhausted' the smaller dataset is
restarted and oversampled, so the epoch bookkeeping is meaningless; under
'all_exhausted_without_replacement' the interleaved length is exactly the sum of the parts.

This test uses two small synthetic datasets rather than the real WorldSpeech configs: the
semantics under test belong to interleave_datasets, not to any particular corpus, and
synthetic data makes the expected counts exact and the test offline and instant. Both the
map-style and the streaming/iterable paths are checked, because the training runs use
streaming=True and the two implementations are separate code in datasets.

Run:  python verify_interleave_semantics.py
Exit code is non-zero if the semantics differ from what the analysis assumes.
"""
import sys

from datasets import Dataset, interleave_datasets

# Deliberately lopsided, in the same spirit as a real pairing where one country-language
# config is much smaller than the other.
SIZE_A = 100
SIZE_B = 25

ok = True


def check(name, cond, detail=''):
    global ok
    ok = ok and bool(cond)
    print(f'[{"PASS" if cond else "FAIL"}] {name}' + (f'  -- {detail}' if detail else ''))


def make_pair(streaming):
    """Two datasets whose examples are tagged by source, so origin is recoverable."""
    a = Dataset.from_dict({'src': ['A'] * SIZE_A, 'i': list(range(SIZE_A))})
    b = Dataset.from_dict({'src': ['B'] * SIZE_B, 'i': list(range(SIZE_B))})
    if streaming:
        return a.to_iterable_dataset(), b.to_iterable_dataset()
    return a, b


def upstream_interleave(datasets_list):
    """The exact call used by the training code path, including its fallback.

    Copied rather than imported: this repo does analysis only and must not depend on the
    training framework, per the repo conventions.
    """
    try:
        return interleave_datasets(
            datasets_list,
            probabilities=[1 / len(datasets_list) for _ in range(len(datasets_list))],
            stopping_strategy='all_exhausted_without_replacement', seed=0
        ), 'all_exhausted_without_replacement'
    except Exception:
        return interleave_datasets(
            datasets_list, stopping_strategy='all_exhausted'), 'all_exhausted (fallback)'


def counts(ds):
    """Total and per-source example counts, by full iteration."""
    n, per = 0, {}
    for row in ds:
        n += 1
        per[row['src']] = per.get(row['src'], 0) + 1
    return n, per


def unique_counts(ds):
    """Distinct (src, i) pairs -- catches duplication that a total count alone would miss."""
    seen = set()
    for row in ds:
        seen.add((row['src'], row['i']))
    return len(seen)


print(f'\nSynthetic pair: A={SIZE_A} examples, B={SIZE_B} examples, '
      f'sum={SIZE_A + SIZE_B}\n')

for streaming in (False, True):
    label = 'streaming / IterableDataset' if streaming else 'map-style / Dataset'
    print(f'--- {label} ---')

    # Baseline: each dataset alone, so the "without interleave" counts are established
    # rather than assumed.
    a, b = make_pair(streaming)
    n_a, _ = counts(a)
    n_b, _ = counts(b)
    check(f'[{label}] dataset A alone yields {SIZE_A}', n_a == SIZE_A, f'got {n_a}')
    check(f'[{label}] dataset B alone yields {SIZE_B}', n_b == SIZE_B, f'got {n_b}')

    # The upstream call.
    a, b = make_pair(streaming)
    inter, strategy_used = upstream_interleave([a, b])
    print(f'      strategy actually used: {strategy_used}')
    n_i, per = counts(inter)

    check(f'[{label}] upstream path uses all_exhausted_without_replacement '
          f'(not the bare-except fallback)',
          strategy_used == 'all_exhausted_without_replacement', strategy_used)
    check(f'[{label}] interleaved total == sum of parts ({SIZE_A + SIZE_B})',
          n_i == SIZE_A + SIZE_B, f'got {n_i}')
    check(f'[{label}] every A example appears exactly once',
          per.get('A') == SIZE_A, f'got {per.get("A")}')
    check(f'[{label}] every B example appears exactly once (no oversampling)',
          per.get('B') == SIZE_B, f'got {per.get("B")}')

    a, b = make_pair(streaming)
    inter, _ = upstream_interleave([a, b])
    n_u = unique_counts(inter)
    check(f'[{label}] no duplicated examples ({SIZE_A + SIZE_B} distinct)',
          n_u == SIZE_A + SIZE_B, f'got {n_u} distinct')

    # Contrast: plain all_exhausted WITH uniform probabilities is the behaviour the analysis
    # originally assumed. Showing it differs is what makes the check above meaningful.
    a, b = make_pair(streaming)
    plain = interleave_datasets([a, b], probabilities=[0.5, 0.5],
                                stopping_strategy='all_exhausted', seed=0)
    n_p, per_p = counts(plain)
    check(f'[{label}] plain all_exhausted DOES oversample B '
          f'(the behaviour we are NOT using)',
          per_p.get('B', 0) > SIZE_B,
          f'B seen {per_p.get("B")}x vs {SIZE_B} available, total {n_p}')

    print()

# A three-way pairing, since nothing restricts the config list to two entries.
print('--- three-way, streaming ---')
sizes = (60, 20, 5)
ds_list = [Dataset.from_dict({'src': [c] * n, 'i': list(range(n))}).to_iterable_dataset()
           for c, n in zip('ABC', sizes)]
inter, strategy_used = upstream_interleave(ds_list)
n_i, per = counts(inter)
check(f'three-way total == sum of parts ({sum(sizes)})', n_i == sum(sizes), f'got {n_i}')
check('three-way per-source counts are exact',
      [per.get(c) for c in 'ABC'] == list(sizes), f'got {[per.get(c) for c in "ABC"]}')

print(f'\n{"ALL TESTS PASSED" if ok else "SOME TESTS FAILED"}')
sys.exit(0 if ok else 1)
