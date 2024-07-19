import logging

import pandas as pd
import numpy as np
import sys
import argparse
import webbrowser
import tempfile
from typing import List, Union
import random

"""

If you have a .csv file that you want to sample 'stratified' based on the first numerical column, do:

$ cat some_csv_file.csv | strample 

Another example:

$ strample some_csv_file.csv --descending --quantiles 30 --key score

For various options (sort key, sample size, number of quantiles, ascending vs. descending) see the help.

"""

# TODO: Include basic stats, histogram etc? Integrate with pandas-profiling? And/or streamlit?
# TODO: Allow span highlights (e.g., for QA)?
# TODO: Allow more flexible quantile specification, e.g., 1,4,90,4,1
# TODO: Write seed to html.

def parse_arguments():
    parser = argparse.ArgumentParser(description='For .csv data, based on a numerical column --key, generate a simple HTML page with a sample per quantile.')
    parser.add_argument('csv', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                        help='Input CSV file or stdin')
    parser.add_argument('-k', '--key', type=str, default=None, help='Which value to sort, color and stratify by; first numerical column by default.')
    parser.add_argument('-n', '--sample_size', type=int, default=20, help='Number of rows per sub-table')
    parser.add_argument('-q', '--quantiles', type=int, default=10, help='Number of quantiles')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--out', type=argparse.FileType('w'), default=None, help='Output file for .html; otherwise served in webbrowser.')
    parser.add_argument('--descending', action='store_true', required=False, help='To show samples in descending order.')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    if args.seed is None:
        args.seed = random.randint(0, 9999)

    logging.basicConfig(level=logging.INFO)
    logging.info(f'Random seed: {args.seed}')
    np.random.seed(args.seed)

    data = pd.read_csv(args.csv)

    if args.key is None:
        args.key = get_first_numerical_column(data)

    html_output = generate_html(data, args.sample_size, args.quantiles, args.key, not args.descending)

    if args.out:
        args.out.write(html_output)
    else:
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
            url = 'file://' + f.name
            f.write(html_output)
        webbrowser.open(url)


def get_first_numerical_column(data: pd.DataFrame) -> Union[None, int]:
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    if not any(numerical_columns):
        return None

    return numerical_columns[0]


def bottom_html(data: pd.DataFrame, sample_size: int, key: str) -> List[str]:
    html = [
        f"<h2>Bottom {sample_size} rows</h2>",
        data.nsmallest(sample_size, key).sort_values(by=key).to_html(index=False)
    ]
    return html


def top_html(data: pd.DataFrame, sample_size: int, key: str) -> List[str]:
    html = [
        f"<h2>Top {sample_size} rows</h2>",
        data.nlargest(sample_size, key).sort_values(by=key).to_html(index=False)
    ]
    return html


def quantiles_html(data: pd.DataFrame, num_quantiles: int, sample_size: int, key: str, ascending: bool) -> List[str]:
    html = []

    quantiles = np.linspace(0, 1, num_quantiles + 1)
    for i in range(num_quantiles) if ascending else range(num_quantiles-1, -1, -1):
        q_start = quantiles[i]
        q_end = quantiles[i + 1]
        quantile_data = data[
            (data[key] > data[key].quantile(q_start)) & (data[key] <= data[key].quantile(q_end))]
        sample_data = quantile_data.sample(min(sample_size, len(quantile_data)), random_state=1).sort_values(by=key)
        if ascending:
            html.append(f"<h2>Quantile {q_start * 100:.0f}-{q_end * 100:.0f}%</h2>")
        else:
            html.append(f"<h2>Quantile {q_end * 100:.0f}-{q_start * 100:.0f}%</h2>")
        html.append(sample_data.to_html(index=False))

    return html


def generate_html(data: pd.DataFrame, sample_size: int, num_quantiles: int, key: str, ascending: bool = True):
    html = ["<html><body>"]

    html.extend(bottom_html(data, sample_size, key) if ascending else top_html(data, sample_size, key))

    html.extend(quantiles_html(data, num_quantiles, sample_size, key, ascending))

    html.extend(top_html(data, sample_size, key) if ascending else bottom_html(data, sample_size, key))

    html.append("</body></html>")
    return "".join(html)


if __name__ == "__main__":
    main()
