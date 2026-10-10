# Cheap Talk at El Farol

Source for the paper *Cheap Talk at El Farol: Independent Deep Q-Learners Learn to Babble, Not to Deceive*.

Every number, table and figure comes from the raw runs:

| File | Made by |
| --- | --- |
| `results.tex` | `analysis.py`: every number quoted in the text |
| `table_contrast.tex`, `table_information.tex` | `analysis.py` |
| `figures/*.pdf` | `analysis.py` |

## Reproduce

From the repository root, with PyTorch, numpy, pandas and matplotlib installed:

```bash
bash paper/run_sweep.sh
```

This runs 72 towns (N = 80, 100, 150 × 12 seeds × silent and broadcast, 100 nights each) on CPU, runs `analyzer.py` on each population size, and regenerates the tables and figures. It takes about 15 minutes with 4 parallel jobs (`JOBS=8 bash paper/run_sweep.sh` for more). Set `PYTHON` if your interpreter isn't `python3`.

## Build the PDF

Upload this folder to Overleaf, or run:

```bash
latexmk -pdf main.tex
```

For arXiv, submit `main.tex`, `refs.bib`, `main.bbl`, `results.tex`, both `table_*.tex` files and `figures/`.
