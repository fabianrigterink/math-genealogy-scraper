# Math Genealogy

Scraper for the [Mathematics Genealogy Project](https://www.mathgenealogy.org/),
plus the analysis pipeline behind the EDA in `analysis/eda.ipynb`.

A copy of the database, scraped on 2019-06-17, is contained in `data.json`
(263k records). You do not need to re-scrape to run the analysis.

## Setup

Python 3.9+ is recommended (the notebook uses modern pandas/plotly).

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`visualize.py` shells out to the Graphviz `dot` binary. On macOS:

```
brew install graphviz
```

## Scraping (optional)

A snapshot is already checked in as `data.json`, so you only need this if you
want to refresh the dataset.

```
python fetch.py
```

The output is a dictionary with a single key `nodes`, mapping to a list of
dictionaries (specified by `parse.py`) of the form

```
{
  "students": [int, int, ...],   # refers to the id field
  "advisors": [int, int, ...],
  "name": str,
  "school": str,
  "subject": str,
  "thesis": str,
  "country": str,
  "year": int,
  "id": int,
}
```

Fields that were not found are null, or the empty list, as appropriate.

Example:

```
{
  "id": 186481,
  "name": "John Anthony Gerard Roberts",
  "thesis": "Order and Chaos in reversible Dynamical Systems",
  "school": "University of Melbourne",
  "country": "Australia",
  "year": 1990,
  "subject": null,
  "advisors": [53185, 116308],
  "students": [186482, 186484, 186486, 186485]
}
```

To be a nice person, the scraper rate-limits itself to 5 concurrent workers.
Downloading the entire database that way takes about 6 hours. Please use the
checked-in `data.json` unless you really need a fresh copy.

## Analysis

The analysis pipeline is two steps: build a clean graph, then either render a
single mathematician's ancestor tree, or regenerate the full EDA notebook.

### 1. Build the graph

```
python analysis/build_graph.py
```

Reads `data.json`, drops implausibly-dated records, removes the handful of
cycles the raw data contains (using a year-based heuristic), and writes
`analysis/graph.pkl` — a `networkx.DiGraph` with 262,967 nodes and 285,398
edges as of the 2019 snapshot.

Options: `--input` (default `data.json`), `--output` (default
`analysis/graph.pkl`).

### 2a. Render one mathematician's ancestor tree

```
python analysis/visualize.py --name "David Hilbert"
```

Writes `analysis/genealogy.html` — a standalone, self-contained interactive
SVG (pan/zoom via `svg-pan-zoom`, hover tooltips, clickable nodes linking
back to MGP).

`--name` accepts an exact match or a unique substring. Ambiguous matches
print a disambiguation list. Other output formats are supported via the
file extension: `--output foo.svg`, `foo.png`, `foo.pdf`, `foo.dot`, etc.

### 2b. Regenerate the EDA notebook

`analysis/eda.ipynb` is generated from `analysis/_build_eda.py` — that
script is the authoritative source. To regenerate after editing cells:

```
python analysis/_build_eda.py
jupyter lab analysis/eda.ipynb
```

The notebook walks through ~30 questions about the genealogy (network
structure, growth, patriarchs, generation length, geography, subjects,
gender, historical ruptures, prize laureates, and so on).

### 2c. Regenerate the blog figures

The 8 PNGs referenced by `analysis/blog/findings_summary.md` are produced
by a standalone script that mirrors the relevant notebook cells, so you
can refresh them without re-executing the full notebook:

```
python analysis/_build_blog_figures.py
```

Output goes to `analysis/blog/figures/`.

## Files

| Path                       | Purpose                                          |
| -------------------------- | ------------------------------------------------ |
| `fetch.py`                 | Scrape MGP into `data.json`                      |
| `parse.py`                 | HTML → record parser used by `fetch.py`          |
| `validate.py`              | Sanity-check `data.json`                         |
| `compress.py`              | Compact `data.json` → `genealogy_graph.json`     |
| `analysis/build_graph.py`  | `data.json` → `analysis/graph.pkl` (clean DAG)   |
| `analysis/visualize.py`    | Render one mathematician's ancestor tree         |
| `analysis/_build_eda.py`   | Source-of-truth script for `analysis/eda.ipynb`  |
| `analysis/eda.ipynb`       | Generated EDA notebook                           |
| `analysis/_build_blog_figures.py` | Regenerates `analysis/blog/figures/*.png` |
| `analysis/blog/`           | Findings summary + figures for the blog post     |
