# strample: Stratified sampling for data exploration #

Sometimes you have a `.csv` file containing items (e.g., texts) each with a given numerical value 
(e.g., some kind of probability or score), and you may want to see some random rows. It can be useful
to divide the data into quantiles (based on the numerical value), and then randomly sample some rows per quantile.
That's what this does.

## Install

```bash
pip install pipx
pipx install git+https://github.com/mwestera/strample
```

## Usage

Basic case:

```bash
$ strample some_csv_file.csv
``` 

With more options:

```bash
$ strample some_csv_file.csv --descending --quantiles 30 --key score
```

For the various options (sort key, sample size, number of quantiles, ascending vs. descending) see the help.
