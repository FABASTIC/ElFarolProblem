# Cheap Talk at El Farol

Source for the paper *Cheap Talk at El Farol: A Free Pre-Play Channel Leaves Independent Deep Q-Learners Unchanged*.

## What generates what

The paper's numbers come from a three-stage pipeline:

1. **`run_sweep.sh`** runs `experiment.py` (the PyTorch engine, repository root) once per town and writes raw per-night and per-agent logs to `outputs/paper/h<NIGHTS>/n<N>/`.
2. **`analyzer.py`** (repository root) computes the statistics for each population size: equilibrium gaps, regret, Gini, strategy drift, and paired silent-versus-broadcast contrasts with bootstrap intervals and sign-flip p-values. `run_sweep.sh` calls it twice per population size, once for all nights (`analytics_report.json`) and once for the last 20% of nights (`analytics_report_late.json`).
3. **`analysis.py`** (this folder) reads those reports and the raw logs and writes everything the paper uses:

| File | Contents |
| --- | --- |
| `results.tex` | Every number quoted in the text, as `\r...` macros |
| `table_contrast_late.tex`, `table_contrast_long.tex`, `table_contrast_short.tex` | Contrast tables for the last 200, all 1000 and the first 100 nights, with BH-FDR q-values within each window |
| `table_information.tex` | Claim–attendance correlations and lie composition |
| `figures/learning.pdf`, `figures/contrast.pdf`, `figures/minds.pdf` | All figures |

No number in `main.tex` is typed by hand. `check_macros.py` fails if the text uses a result macro that `results.tex` does not define.

## Reproduce

See [Reproduce the paper from scratch](../README.md#reproduce-the-paper-from-scratch) in the repository README for the exact commands and runtimes.

## Build the PDF

With a TeX distribution that has `latexmk` and `pdflatex`:

```bash
latexmk -pdf main.tex
```

Or upload this folder to Overleaf.

## arXiv bundle

`arxiv/` holds exactly what arXiv needs: `main.tex`, `refs.bib`, `main.bbl`, `results.tex`, the `table_*.tex` files and `figures/`. It compiles on its own. To rebuild it after changing the paper, run from this folder:

```bash
latexmk -pdf main.tex
rm -rf arxiv && mkdir -p arxiv/figures
cp main.tex refs.bib main.bbl results.tex table_*.tex arxiv/
cp figures/*.pdf arxiv/figures/
```

## License

The paper (text, figures and tables) is licensed under CC BY 4.0; see `LICENSE` in this folder. The code is MIT-licensed; see the repository root.
