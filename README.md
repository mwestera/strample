# rmnl: remove newlines #

Useful for removing superfluous newlines when, e.g., copying text from a pdf.

## Install ##

```bash
sudo apt-get install xclip
pip install git+https://github.com/mwestera/rmnl
```

## Examples ##

```bash
$ rmnl old.txt  #  from a file to stdout
$ rmnl old.txt -p #  same but keep newlines after punctuation
$ rmnl old.txt -d #  same but keep double newlines (as single newlines)
$ rmnl old.txt > new.txt  #  from a file to a new file
$ cat *.txt | rmnl > new.txt  # from stdin
$ rmnl -c # from and to clipboard
$ rmnl -cpd # from and to clipboard, keeping some newlines
```
