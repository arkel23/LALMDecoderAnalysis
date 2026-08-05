"""Transcribes the Tiny Aya report's per-variant language-composition tables into a tidy CSV.

Source: arXiv:2603.11510, Appendix A, Tables 8-14 (English, European, West Asia, South Asia,
Asia Pacific, African, Code).

This turns "specialisation" from a categorical earth/fire/water label into a continuous
variable -- the share of each variant's post-training mix in a given language -- which is what
lets the study ask whether performance tracks how much of a language the decoder actually saw.

Parses the published HTML rather than restating numbers, so the CSV can be regenerated and
diffed. Cached under data/ on first run. Standard library only: pandas.read_html needs lxml,
and adding a dependency would break the bare-checkout bar.

The tables' columns are data MIXES, not variant names. The mapping is stated by the report and
independently CHECKED against the numbers in verify_mix_mapping(), so a revision that moves the
columns fails rather than silently relabelling them.

Usage:
    python fetch_tinyaya_composition.py [--refresh]
"""
import os
import re
import argparse
import urllib.request
from html.parser import HTMLParser

import pandas as pd

ARXIV_ID = '2603.11510v1'
SOURCE_URL = f'https://arxiv.org/html/{ARXIV_ID}'
CACHE_HTML = os.path.join('data', 'tinyaya_report', f'{ARXIV_ID}.html')

# Column header in the report -> the model variant trained on that mix.
MIX_TO_VARIANT = {
    'All Regions': 'global',
    'South Asia': 'fire',
    'Europe+WA+AP': 'water',
    'Europe+WA+Af': 'earth',
}

# Caption prefix -> the region grouping the table covers.
TABLE_REGION = {
    'Table 8': 'English',
    'Table 9': 'Europe',
    'Table 10': 'West Asia',
    'Table 11': 'South Asia',
    'Table 12': 'Asia Pacific',
    'Table 13': 'Africa',
    'Table 14': 'Code',
}

FLOAT_FORMAT = '%.6f'


class _TableParser(HTMLParser):
    """Minimal <table> -> list-of-rows extractor. Stdlib only, deliberately."""

    def __init__(self):
        super().__init__()
        self.rows, self._row, self._cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self._row = []
        elif tag in ('td', 'th'):
            self._cell = []

    def handle_endtag(self, tag):
        if tag == 'tr' and self._row is not None:
            self.rows.append(self._row)
            self._row = None
        elif tag in ('td', 'th') and self._cell is not None:
            text = re.sub(r'\s+', ' ', ''.join(self._cell)).strip()
            if self._row is not None:
                self._row.append(text)
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def strip_tags(fragment):
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', fragment)).strip()


def get_html(refresh=False):
    os.makedirs(os.path.dirname(CACHE_HTML), exist_ok=True)
    if os.path.exists(CACHE_HTML) and not refresh:
        print(f'Using cached {CACHE_HTML} (pass --refresh to re-fetch)')
        return open(CACHE_HTML, encoding='utf-8').read()

    print(f'Fetching {SOURCE_URL} ...')
    req = urllib.request.Request(
        SOURCE_URL, headers={'User-Agent': 'Mozilla/5.0 (LALMDecoderAnalysis; research)'})
    html = urllib.request.urlopen(req, timeout=180).read().decode('utf-8', 'replace')
    with open(CACHE_HTML, 'w', encoding='utf-8') as fh:
        fh.write(html)
    print(f'Cached {len(html)} bytes to {CACHE_HTML}')
    return html


def parse_composition(html):
    rows = []
    for figure in re.findall(r'<figure.*?</figure>', html, re.S):
        cap_match = re.search(r'<figcaption[^>]*>(.*?)</figcaption>', figure, re.S)
        if not cap_match:
            continue
        caption = strip_tags(cap_match.group(1))
        if 'data proportion' not in caption:
            continue

        table_no = re.match(r'(Table\s*\d+)', caption)
        table_no = re.sub(r'\s+', ' ', table_no.group(1)) if table_no else None
        region = TABLE_REGION.get(table_no)

        parser = _TableParser()
        parser.feed(figure)
        if not parser.rows:
            continue

        header, *body = parser.rows
        mixes = header[1:]
        for row in body:
            if len(row) != len(header) or not row[0]:
                continue
            for mix, value in zip(mixes, row[1:]):
                try:
                    pct = float(value)
                except ValueError:
                    continue
                rows.append({
                    'language': row[0],
                    # Each regional table ends with a 'Subtotal' row. Treating it as a
                    # language double-counts the whole region -- it inflated every mix total
                    # to ~180%. Kept as a separate row_type because it is the strongest
                    # available cross-check: a region's languages must sum to its subtotal.
                    'row_type': 'subtotal' if row[0].lower().startswith('subtotal')
                                else 'language',
                    'report_region': region,
                    'report_table': table_no,
                    'mix': mix,
                    'variant': MIX_TO_VARIANT.get(mix),
                    'proportion_pct': pct,
                })

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit('No composition tables parsed -- the report layout may have changed.')
    return out


def verify_mix_mapping(df):
    """Check the column->variant mapping against the numbers, not just against the prose.

    Each regional mix must give its own region the largest share, and must give a region it
    excludes a near-zero share. If the report is revised and the columns are reordered, this
    fails loudly instead of silently relabelling every proportion in the CSV.
    """
    ok = True

    def mean_for(region, variant):
        sel = df[(df['report_region'] == region) & (df['variant'] == variant)]
        return sel['proportion_pct'].mean()

    checks = [
        ('African languages get their largest share from the earth mix',
         mean_for('Africa', 'earth') > max(mean_for('Africa', v)
                                           for v in ('fire', 'water', 'global'))),
        ('African languages are ~absent from the water mix', mean_for('Africa', 'water') < 0.2),
        ('South Asian languages get their largest share from the fire mix',
         mean_for('South Asia', 'fire') > max(mean_for('South Asia', v)
                                              for v in ('earth', 'water', 'global'))),
        ('Asia-Pacific languages get their largest share from the water mix',
         mean_for('Asia Pacific', 'water') > max(mean_for('Asia Pacific', v)
                                                 for v in ('earth', 'fire', 'global'))),
        ('Asia-Pacific languages are ~absent from the earth mix',
         mean_for('Asia Pacific', 'earth') < 0.2),
        ('every mix maps to a known variant', df['variant'].notna().all()),
    ]
    for name, cond in checks:
        ok = ok and bool(cond)
        print(f'  [{"PASS" if cond else "FAIL"}] {name}')

    # Each region's languages must sum to that region's printed Subtotal. This is the
    # strongest check available: it validates the parse row by row against a number the
    # report itself computed.
    print('\n  per-region language sums vs the report\'s printed Subtotal:')
    langs = df[df['row_type'] == 'language']
    subs = df[df['row_type'] == 'subtotal']
    for (region, mix), g in subs.groupby(['report_region', 'mix']):
        printed = g['proportion_pct'].iloc[0]
        summed = langs[(langs['report_region'] == region)
                       & (langs['mix'] == mix)]['proportion_pct'].sum()
        close = abs(printed - summed) <= 0.35        # printed values are 1 d.p.
        ok = ok and close
        if not close:
            print(f'    [FAIL] {region:13s} {mix:14s} languages={summed:6.1f} '
                  f'vs printed subtotal={printed:6.1f}')
    print('    all regions match their printed subtotals'
          if ok else '    MISMATCH above')

    # And English + Code + the regional subtotals must account for ~100% of each mix.
    print('\n  mix totals (English + Code + regional subtotals, should be ~100%):')
    for mix in sorted(df['mix'].unique()):
        tot = (subs[subs['mix'] == mix]['proportion_pct'].sum()
               + langs[(langs['mix'] == mix)
                       & (langs['report_region'].isin(['English', 'Code']))]
               ['proportion_pct'].sum())
        good = 97 <= tot <= 103
        ok = ok and good
        print(f'    {mix:15s} {tot:6.1f}  {"ok" if good else "OUT OF RANGE"}')

    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--refresh', action='store_true', help='Force re-download of the report.')
    p.add_argument('--output_file', type=str,
                   default=os.path.join('data', 'tinyaya_report',
                                        'tinyaya_language_composition.csv'))
    p.add_argument('--wide_file', type=str,
                   default=os.path.join('data', 'tinyaya_report',
                                        'tinyaya_language_composition_wide.csv'))
    args = p.parse_args()

    df = parse_composition(get_html(args.refresh))
    df['source_url'] = SOURCE_URL

    n_lang = df[df['row_type'] == 'language']['language'].nunique()
    print(f'\nParsed {len(df)} rows from {df["report_table"].nunique()} tables: '
          f'{n_lang} languages plus per-region subtotals.\n')
    print('Verifying the mix-to-variant mapping against the numbers:')
    ok = verify_mix_mapping(df)

    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    df.to_csv(args.output_file, index=False, float_format=FLOAT_FORMAT)
    print(f'\nWrote {len(df)} rows to {args.output_file}')

    wide = (df[df['row_type'] == 'language']
            .pivot_table(index=['language', 'report_region'], columns='variant',
                         values='proportion_pct').reset_index())
    wide.to_csv(args.wide_file, index=False, float_format=FLOAT_FORMAT)
    print(f'Wrote {len(wide)} rows to {args.wide_file}')

    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
