import logging

import pandas as pd
import numpy as np
import sys
import argparse
import webbrowser
import tempfile
from typing import List, Union
import random
import functools
import itertools
import json
import io

"""

If you have a .csv file that you want to sample 'stratified' based on the first numerical column, do:

$ cat some_csv_file.csv | strample 

Another example:

$ strample some_csv_file.csv --descending --quantiles 30 --key score

For various options (sort key, sample size, number of quantiles, ascending vs. descending, per-token coloration) see the help.

"""

# TODO: Include basic stats, histogram etc? Integrate with pandas-profiling? And/or streamlit?
# TODO: Allow more flexible quantile specification, e.g., 1,4,90,4,1

def parse_arguments():
    parser = argparse.ArgumentParser(description='For .csv data, based on a numerical column --key, generate a simple HTML page with a sample per quantile.')
    parser.add_argument('file', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help='Input csv or json file or stdin')
    parser.add_argument('-k', '--key', type=str, default=None, help='Which value to sort, color and stratify by; first numerical column by default.')
    parser.add_argument('-n', '--sample_size', type=int, default=20, help='Number of rows per sub-table')
    parser.add_argument('-q', '--quantiles', type=int, default=10, help='Number of quantiles')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--out', type=argparse.FileType('w'), default=None, help='Output file for .html; otherwise served in webbrowser.')
    parser.add_argument('--descending', action='store_true', required=False, help='To show samples in descending order.')
    parser.add_argument('-s', '--span', type=str, required=False, help='To highlight a single span, a csv triple of field names: text,start,end')
    parser.add_argument('--tokens', type=str, required=False, help='To highlight individual tokens by score, a csv triple of field names: text,token_scores,token_spans')
    args = parser.parse_args()
    if args.span:
        text, start, end = args.span.split(',') # TODO make proper csv
        args.span = (text, start, end)
    if args.tokens:
        text, scores, spans = args.tokens.split(',')
        args.tokens = (text, scores, spans)
    return args


def main():
    args = parse_arguments()

    if args.seed is None:
        args.seed = random.randint(0, 9999)

    logging.basicConfig(level=logging.INFO)
    logging.info(f'Random seed: {args.seed}')
    np.random.seed(args.seed)

    make_html = functools.partial(generate_html,
                                  sample_size=args.sample_size,
                                  num_quantiles=args.quantiles,
                                  key=args.key,
                                  ascending=not args.descending,
                                  span=args.span,
                                  tokens=args.tokens,
                                  seed=args.seed)

    file, is_jsonl = peek_if_jsonl(args.file)
    data = pd.read_json(file, orient='records', lines=True) if is_jsonl else pd.read_csv(file)

    if args.key is None:
        args.key = get_first_numerical_column(data)

    html_output = make_html(data)

    if args.out:
        args.out.write(html_output)
    else:
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
            url = 'file://' + f.name
            f.write(html_output)
        webbrowser.open(url)


def peek_if_jsonl(file):
    file, peekfile = itertools.tee(file)
    firstline = next(peekfile)
    file = StringIteratorIO(file)
    try:
        d = json.loads(firstline.strip())
        if isinstance(d, dict):
            return file, True
    except json.JSONDecodeError:
        pass
    return file, False


def get_first_numerical_column(data: pd.DataFrame) -> Union[None, int]:
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    if not any(numerical_columns):
        return None

    return numerical_columns[0]


def bottom_html(data: pd.DataFrame, sample_size: int, key: str, ascending: bool, span: tuple, tokens: tuple) -> List[str]:
    sample = data.nsmallest(sample_size, key).sort_values(by=key, ascending=ascending)
    html = [
        f"<h2>Bottom {sample_size} rows</h2>",
        sample_to_html(sample, span, tokens)
    ]
    return html


def top_html(data: pd.DataFrame, sample_size: int, key: str, ascending: bool, span: tuple, tokens: tuple) -> List[str]:
    sample = data.nlargest(sample_size, key).sort_values(by=key, ascending=ascending)
    html = [
        f"<h2>Top {sample_size} rows</h2>",
        sample_to_html(sample, span, tokens),
    ]
    return html


def quantiles_html(data: pd.DataFrame, num_quantiles: int, sample_size: int, key: str, ascending: bool, span: tuple, tokens: tuple) -> List[str]:
    html = []

    quantiles = np.linspace(0, 1, num_quantiles + 1)
    for i in range(num_quantiles) if ascending else range(num_quantiles-1, -1, -1):
        q_start = quantiles[i]
        q_end = quantiles[i + 1]
        quantile_data = data[(data[key] > data[key].quantile(q_start)) & (data[key] <= data[key].quantile(q_end))]
        sample = quantile_data.sample(min(sample_size, len(quantile_data)), random_state=1).sort_values(by=key, ascending=ascending)
        if ascending:
            html.append(f"<h2>Quantile {q_start * 100:.0f}-{q_end * 100:.0f}%</h2>")
        else:
            html.append(f"<h2>Quantile {q_end * 100:.0f}-{q_start * 100:.0f}%</h2>")

        html.append(sample_to_html(sample, span, tokens))

    return html


def colormap(score):
    r = 'f'
    g = 'f'
    b = hex(round((1-score) * 15))[-1]
    return f'#{r}{g}{b}'


def sample_to_html(sample: pd.DataFrame, span: tuple, tokens: tuple):
    for i, row in sample.iterrows():
        markers = {}
        if span:
            spanmarkers = markers.setdefault(span[0], [])
            spanmarkers.append((row[span[1]], '<u>'))
            spanmarkers.append((row[span[2]], '</u>'))
        if tokens:
            tokenmarkers = markers.setdefault(tokens[0], [])
            for score, (start, end) in zip(row[tokens[1]], row[tokens[2]]):
                tokenmarkers.append((start, f'<span style="background-color:{colormap(score)};">'))
                tokenmarkers.append((end, f'</span>'))
        for text_column, markers in markers.items():
            text = row[text_column]
            marked_text = []
            markers = [(0, '')] + markers + [(len(text), '')]
            markers.sort(reverse=True)
            for (prev_offset, _), (offset, marker) in zip(markers, markers[1:]):
                marked_text.append(text[offset:prev_offset])
                marked_text.append(marker)
            sample.at[i, text_column] = ''.join(reversed(marked_text))
    columns_to_remove = []
    if span:
        columns_to_remove += [span[1], span[2]]
    if tokens:
        columns_to_remove += [tokens[1], tokens[2]]
    sample = sample[[col for col in sample.columns if col not in columns_to_remove]]
    html = sample.to_html(index=False).replace('&lt;', '<').replace('&gt;', '>')
    return html


def generate_html(data: pd.DataFrame, sample_size: int, num_quantiles: int, key: str, ascending: bool = True, span: tuple = None, tokens: tuple = None, seed=None):
    html = [f"<html><body><h1>Strample: Random samples stratified by {key} (total {len(data)} rows; seed: {seed})</h1>"]

    bottom = bottom_html(data, sample_size, key, ascending, span, tokens)
    top = top_html(data, sample_size, key, ascending, span, tokens)
    html.extend(bottom if ascending else top)
    html.extend(quantiles_html(data, num_quantiles, sample_size, key, ascending, span, tokens))
    html.extend(top if ascending else bottom)

    html.append("</body></html>")
    return "".join(html)


class StringIteratorIO(io.TextIOBase):
    """
    https://stackoverflow.com/questions/12593576/adapt-an-iterator-to-behave-like-a-file-like-object-in-python
    """

    def __init__(self, iter):
        self._iter = iter
        self._left = ''

    def readable(self):
        return True

    def _read1(self, n=None):
        while not self._left:
            try:
                self._left = next(self._iter)
            except StopIteration:
                break
        ret = self._left[:n]
        self._left = self._left[len(ret):]
        return ret

    def read(self, n=None):
        l = []
        if n is None or n < 0:
            while True:
                m = self._read1()
                if not m:
                    break
                l.append(m)
        else:
            while n > 0:
                m = self._read1(n)
                if not m:
                    break
                n -= len(m)
                l.append(m)
        return ''.join(l)

    def readline(self):
        l = []
        while True:
            i = self._left.find('\n')
            if i == -1:
                l.append(self._left)
                try:
                    self._left = next(self._iter)
                except StopIteration:
                    self._left = ''
                    break
            else:
                l.append(self._left[:i+1])
                self._left = self._left[i+1:]
                break
        return ''.join(l)


if __name__ == "__main__":
    main()
