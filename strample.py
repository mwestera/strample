import logging

import pandas as pd
import numpy as np
import sys
import argparse
import webbrowser
import tempfile
from typing import List, Union, Tuple
import random
import functools
import itertools
import json
import io

import seaborn
import matplotlib.pyplot as plt
import base64

"""

If you have a .csv file that you want to sample 'stratified' based on the first numerical column, do:

$ cat some_csv_file.csv | strample 

Another example:

$ strample some_csv_file.csv --descending --quantiles 30 --key score

In addition to html, you can also output the sampled data as csv, with added column 'strample' recording the quantile each row comes from:

$ strample some_csv_file.csv --descending --quantiles 30 --key score --csv_out make_this_file_too.csv

For various options (sort key, sample size, number of quantiles, ascending vs. descending, per-token coloration) see the help.

"""

N_DECIMALS = 5

# TODO: Include basic stats, histogram etc? Integrate with pandas-profiling? And/or streamlit?
# TODO: Allow more flexible quantile specification, e.g., 1,4,90,4,1

def parse_arguments():
    global N_DECIMALS

    parser = argparse.ArgumentParser(description='For .csv data, based on a numerical column --key, generate a simple HTML page with a sample per quantile.')
    parser.add_argument('file', nargs='?', type=argparse.FileType('r'), default=sys.stdin, help='Input csv or json file or stdin')
    parser.add_argument('--title', type=str, default=None, help='String to use in the header. Will use filename if provided.')
    parser.add_argument('-c', '--column', type=str, default=None, help='Which value to sort, color and stratify by; first numerical column by default.')
    parser.add_argument('-n', '--sample_size', type=int, default=20, help='Number of rows per sub-table')
    parser.add_argument('-q', '--quantiles', nargs='+', type=int, default=[10], help='Number of quantiles; or space-separated list of quantile boundaries in percentage')
    parser.add_argument('-d', '--decimals', type=int, default=5, help='Number of decimals for float values')
    # parser.add_argument('-c', '--colorize', action='store_true', required=False, help='To colormap the key column.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--out', type=argparse.FileType('w'), default=None, help='Output file for .html; otherwise served in webbrowser.')
    parser.add_argument('--csv_out', type=argparse.FileType('w'), default=None, help='Output file for .csv; otherwise no csv is generated.')
    parser.add_argument('--descending', action='store_true', required=False, help='To show samples in descending order.')
    parser.add_argument('-s', '--span', type=str, required=False, help='To highlight a single span, a csv triple of field names: text,start,end')
    parser.add_argument('--tokens', type=str, required=False, help='To highlight individual tokens by score, a csv triple of field names: text,token_scores,token_spans')
    parser.add_argument('-k', '--key', type=str, default=None, help='[To be removed; use --column instead.] Which value to sort, color and stratify by; first numerical column by default.')
    args = parser.parse_args()
    if args.key and not args.column:
        args.column = args.key
    if args.span:
        text, start, end = args.span.split(',') # TODO make proper csv
        args.span = (text, start, end)
    if args.tokens:
        text, scores, spans = args.tokens.split(',')
        args.tokens = (text, scores, spans)
    if args.file != sys.stdin and not args.title:
        args.title = args.file.name
    if len(args.quantiles) == 1:
        args.quantiles = np.linspace(0, 1, args.quantiles[0] + 1)
    else:
        args.quantiles = [q / 100 for q in args.quantiles]
        if args.quantiles[0] != 0:
            args.quantiles.insert(0, 0)
        if args.quantiles[-1] != 1:
            args.quantiles.append(1)
    if args.out:
        logging.warning('There\'s something wrong with encoding in output html. Consider using without --out and then manual saving.')
    N_DECIMALS = args.decimals
    return args


def main():
    args = parse_arguments()

    if args.seed is None:
        args.seed = random.randint(0, 9999)

    logging.basicConfig(level=logging.INFO)
    logging.info(f'Random seed: {args.seed}')

    file, is_jsonl = peek_if_jsonl(args.file)
    data = pd.read_json(file, orient='records', lines=True) if is_jsonl else pd.read_csv(file)

    if args.key is None:
        args.key = get_first_numerical_column(data)

    quantile_data = make_samples(data, args.quantiles, args.sample_size, args.key, not args.descending, seed=args.seed)

    write_html(
        quantile_data=quantile_data,
        data=data,
        sample_size=args.sample_size,
        key=args.key,
        ascending=not args.descending,
        span=args.span,
        tokens=args.tokens,
        seed=args.seed,
        outfile=args.out,
        colorize=True,
        title=args.title,
    )

    if args.csv_out:
        write_csv(quantile_data, ascending=not args.descending, outfile=args.csv_out)


def write_csv(quantile_data, ascending, outfile):
    quantiles, top, bottom = quantile_data
    if ascending:
        bottom, top = top, bottom
    samples = [top] + [sample for _, _, sample in quantiles] + [bottom]
    concatenated = pd.concat(samples)
    concatenated.to_csv(outfile)


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


def get_first_numerical_column(data: pd.DataFrame) -> Union[None, str]:
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    if not any(numerical_columns):
        return None

    return numerical_columns[0]


def make_samples(data: pd.DataFrame, quantiles: list[float], sample_size: int, key: str, ascending: bool, seed: int = None) -> Tuple[List[Tuple[float, float, pd.DataFrame]], pd.DataFrame, pd.DataFrame]:

    if seed:
        np.random.seed(seed)

    top = top_sample(data, sample_size, key, ascending)
    quans = quantile_samples(data, quantiles, sample_size, key, ascending)
    bot = bottom_sample(data, sample_size, key, ascending)

    return quans, top, bot


def top_sample(data: pd.DataFrame, sample_size: int, key: str, ascending: bool):
    sample = data.nlargest(sample_size, key).sort_values(by=key, ascending=ascending)
    sample['strample'] = f'top'
    return sample


def bottom_sample(data: pd.DataFrame, sample_size: int, key: str, ascending: bool):
    sample = data.nsmallest(sample_size, key).sort_values(by=key, ascending=ascending)
    sample['strample'] = f'bottom'
    return sample


def quantile_samples(data: pd.DataFrame, quantiles: list[float], sample_size: int, key: str, ascending: bool) -> List[Tuple[float, float, pd.DataFrame]]:
    samples = []

    for i in range(len(quantiles) - 1) if ascending else range(len(quantiles)-1, -1, -1):
        q_start = quantiles[i]
        q_end = quantiles[i + 1]
        quantile_data = data[(data[key] > data[key].quantile(q_start)) & (data[key] <= data[key].quantile(q_end))]
        sample = quantile_data.sample(min(sample_size, len(quantile_data)), random_state=1).sort_values(by=key, ascending=ascending)
        sample['strample'] = f'quantile_{i}'
        samples.append((float(q_start), float(q_end), sample))

    return samples


def colormap_yellow(score):
    r = 'f'
    g = 'f'
    b = hex(round((1-score) * 15))[-1]
    return f'#{r}{g}{b}'


def colormap_redgreen(score, minimum, maximum):

    score = ((score - minimum) / (maximum - minimum))

    score = max(0, min(1, score))

    red = round((1 - score) * 255)
    blue = round(score * 255)
    
    return f'#{red:02x}00{blue:02x}55'


def sample_to_html(sample: pd.DataFrame, span: tuple, tokens: tuple, hue_col=None, hue_min=None, hue_max=None):
    sample = sample[[col for col in sample.columns if col != 'strample']]
    float_format = f'{{:.{N_DECIMALS}f}}'.format

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

        if hue_col is not None:
            bgcolor = colormap_redgreen(row[hue_col], hue_min, hue_max)
            sample.at[i, f'{hue_col}_str'] = f'<td style="background-color:{bgcolor};">{float_format(row[hue_col])}</td>'

    if hue_col is not None:
        sample[hue_col] = sample[f'{hue_col}_str']
        del sample[f'{hue_col}_str']

    columns_to_remove = []
    if span:
        columns_to_remove += [span[1], span[2]]
    if tokens:
        columns_to_remove += [tokens[1], tokens[2]]
    sample = sample[[col for col in sample.columns if col not in columns_to_remove]]
    html = sample.to_html(index=False, float_format=float_format).replace('&lt;', '<').replace('&gt;', '>')
    html = html.replace('<td><td', '<td').replace('</td></td>', '</td>')
    html = html.replace('<tr style="text-align: right;">', '<tr style="text-align: left;">')
    html = html.replace('<table', '<table style="width: 100%;"').replace('<th>rating</th>', f'<th style="width: {40 + N_DECIMALS * 10}pt;">rating</th>')
    return html


def write_html(quantile_data: Tuple, data: pd.DataFrame, sample_size: int, key: str, ascending: bool = True, span: tuple = None, tokens: tuple = None, seed=None, outfile=None, colorize=False, title=None):

    quantile_data, top_data, bottom_data = quantile_data
    colorize_kwargs = {'hue_col': key, 'hue_min': data[key].min(), 'hue_max': data[key].max()} if colorize else {}

    title = title or 'Strample'

    html = [f"<html><body><h1>{title}</h1><h3>Total {len(data)} rows; seed: {seed}.</h3>"]

    seaborn.displot(data=data, x=key, kind='kde', common_norm=False)
    html_img = plot_to_html()
    html.append(html_img)

    bottom_html = [
        f"<h2>Bottom {sample_size} rows</h2>",
        sample_to_html(bottom_data, span, tokens, **colorize_kwargs)
    ]

    top_html = [
        f"<h2>Top {sample_size} rows</h2>",
        sample_to_html(top_data, span, tokens, **colorize_kwargs)
    ]

    html.extend(bottom_html if ascending else top_html)

    for q_start, q_end, sample in quantile_data:
        if ascending:
            html.append(f"<h2>Quantile {q_start * 100:.0f}-{q_end * 100:.0f}%</h2>")
        else:
            html.append(f"<h2>Quantile {q_end * 100:.0f}-{q_start * 100:.0f}%</h2>")
        html.append(sample_to_html(sample, span, tokens, **colorize_kwargs))

    html.extend(top_html if ascending else bottom_html)

    html.append("</body></html>")
    html_code = "".join(html)

    if outfile:
       outfile.write(html_code)
    else:
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
            url = 'file://' + f.name
            f.write(html_code)
        webbrowser.open(url)


def plot_to_html():
    """
    https://stackoverflow.com/a/63381737
    """
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    plt.close()
    plot_base64 = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return f'<img src="data:image/png;base64,{plot_base64}">'


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
