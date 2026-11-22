# Graphical lasso on stock returns

MATH 168 (Networks) project. We estimate a sparse precision matrix Θ for the
returns of 30 large-cap US stocks with the graphical lasso
(Friedman, Hastie & Tibshirani, 2008). Each nonzero off-diagonal entry Θᵢⱼ is an
edge: stocks i and j are still correlated after accounting for all the other
stocks.

This repo contains:
- a **data pipeline**: download prices, compute returns, and review the data
  with plots and summary statistics
- a **baseline**: scikit-learn's graphical lasso on the raw data, used as the
  reference for our own implementation
- **network plots** with fixed node positions, so graphs from different
  datasets or methods can be compared directly

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/ploegs/math168_glasso.git
cd math168_glasso
pip install -r requirements.txt
```

**Run every command from the repo root.** Imports are absolute
(`glasso_project.data_pipeline...`), so scripts run as modules with `python -m`.
Running a file directly (`python plot.py`) fails with
`ModuleNotFoundError: glasso_project`.

## Quick start

The repo already contains both datasets and their outputs, so you can open the
plots in `glasso_project/data/` without running anything. To regenerate them:

```bash
# analyze + plot an existing dataset (no download)
python -m glasso_project.data_pipeline.data_review --interval 1wk --start 2005-01-01 --skip-download

# sklearn glasso baseline (alpha chosen by cross-validation)
python -m glasso_project.baseline_glasso 1wk_2005_2026
```

## Pipeline

Each step reads the previous step's files from `glasso_project/data/<dataset>/`,
so steps can run on their own.

| Step | Command | Reads | Writes |
|---|---|---|---|
| 1. Download | `python -m glasso_project.data_pipeline.download` | Yahoo Finance | `prices.csv`, `meta.json` |
| 2. Analyze | `python -m glasso_project.data_pipeline.analyze <dataset>` | `prices.csv` | `returns.csv`, `summary_stats.csv` |
| 3. Plot | `python -m glasso_project.data_pipeline.plot <dataset>` | the above | `plots/01–07*.png` |
| All three | `python -m glasso_project.data_pipeline.data_review [...]` | | |

A dataset is named `<interval>_<startyear>_<endyear>`, for example
`1wk_2005_2026`. Useful options:

```bash
# download
--interval 1d|1wk|1mo       --start 2005-01-01   --end 2026-11-20
--tickers AAPL MSFT ...      --name my_dataset    --force   # overwrite existing
# analyze
--simple                     # simple returns instead of log returns
--max-missing 0.05           # drop tickers missing more than this fraction of data
# plot
--only 1 5                   # make only figures 01 and 05
--window 52                  # rolling window in periods (default 60 / 26 / 12 for 1d / 1wk / 1mo)
```

Each module also documents its options in `--help`.

### Data-review figures (`data/<dataset>/plots/`)

| Figure | Shows |
|---|---|
| `01_prices` | Prices rebased to 100, log scale, colored by sector |
| `02_returns_timeseries` | Returns over time and cross-sectional rolling volatility |
| `03_distributions` | Pooled standardized returns vs N(0,1), normal Q-Q plot, excess kurtosis per stock |
| `04_summary_stats` | Annualized return, volatility, kurtosis per stock |
| `05_correlation` | Sample correlation matrix, ordered by sector |
| `06_eigenspectrum` | Correlation eigenvalues with the Marchenko–Pastur noise band; loadings of the top 3 eigenvectors |
| `07_rolling_correlation` | Average pairwise correlation over a rolling window |

## Baseline: scikit-learn graphical lasso

```bash
python -m glasso_project.baseline_glasso <dataset>                  # alpha by 5-fold CV
python -m glasso_project.baseline_glasso <dataset> --alpha 0.3      # fixed alpha
python -m glasso_project.baseline_glasso <dataset> --show-min 0.1   # hide weak edges in the network plot only
```

Preprocessing is deliberately minimal. Returns are z-scored, so the input S is
the sample correlation matrix and α is unit-free. There is no factor removal,
no rank transform, and the full sample is used.

**Choosing α.** For any fixed α the algorithm solves

  Θ̂ = argmax over Θ ≻ 0 of  log det Θ − tr(SΘ) − α Σ_{i≠j} |Θᵢⱼ|

α itself is a tuning parameter. By default it is chosen by 5-fold
cross-validation with contiguous time blocks, maximizing the held-out Gaussian
log-likelihood. Cross-validation favors prediction, so it tends to keep many
weak edges; use `--alpha` for a sparser graph.

Outputs go to `data/<dataset>/glasso_sklearn/`:

| File | Contents |
|---|---|
| `precision.csv` | Θ̂, the sparse precision matrix |
| `partial_corr.csv` | Partial correlations ρᵢⱼ = −Θᵢⱼ / √(ΘᵢᵢΘⱼⱼ) (same zero pattern as Θ̂) |
| `edges.csv` | Nonzero edges sorted by \|ρ\|, flagged within-sector or not |
| `path.csv` | Edge count, within-sector share, and CV score for each α on the grid |
| `summary.json` | Chosen α, edge counts, and solver checks: KKT violation and ‖WΘ − I‖ |
| `plots/01_path.png` | CV curve and regularization path |
| `plots/02_corr_vs_partial.png` | Sample correlation vs glasso partial correlation |
| `plots/03_network_{ring,clusters}.png` | Network on the fixed layout |

**Differences from Friedman et al.'s penalty.** sklearn penalizes only the
off-diagonal entries of Θ. The original algorithm also penalizes the diagonal:
it starts from W = S + ρI, so Ŵᵢᵢ = Sᵢᵢ + ρ. For the same α the off-diagonal
structure is comparable, but the diagonals differ.

**Solver tolerance.** We set `enet_tol=1e-7`. With sklearn's default (1e-4),
the solver stalls short of the optimum at some α, which shows up as a
`ConvergenceWarning` and KKT violations around 3e-3.

## Network plots

`network_plot.draw_network` draws every graph on the same layout and scale:
- **Fixed positions.** Each ticker's position depends only on its sector and
  its index in `config.SECTORS`, never on the data. A dropped ticker leaves an
  empty grey circle; the other nodes don't move.
- **Absolute edge scale.** Width and opacity depend on |ρ| on a fixed scale, so
  the same partial correlation looks the same in every plot. Red edges are
  positive, blue negative.

```python
from glasso_project.network_plot import draw_network

draw_network(edges_df, layout="clusters", title="my glasso, α=0.2", fname="net.png")
draw_network(partial_corr_df, layout="ring", ax=axes[0])   # or a matrix, drawn into a subplot
```

Layouts: `clusters` groups each sector, so within-sector edges stay inside a
group and cross-sector edges run through the middle. `ring` puts every ticker
on one circle.

## Repository layout

```
glasso_project/
├── data_pipeline/
│   ├── config.py          # tickers, sectors, paths, periods per year
│   ├── download.py        # step 1
│   ├── analyze.py         # step 2
│   ├── plot.py            # step 3
│   └── data_review.py     # runs steps 1-3
├── baseline_glasso.py     # sklearn reference fit
├── network_plot.py        # standardized network drawing
├── glasso_implementation/ # our implementation (in progress)
└── data/
    ├── 1d_2019_2026/      # daily, 2019-01-03 → 2026-11-19 (n=1982)
    └── 1wk_2005_2026/     # weekly, 2005-01-08 → 2026-11-14 (n=1141)
```

**Universe.** 30 stocks in 7 sectors, defined in `data_pipeline/config.py`:
Tech (AAPL MSFT NVDA ORCL INTC), Financials (JPM BAC GS MS WFC), Energy
(XOM CVX COP SLB), Health (JNJ PFE MRK UNH ABT), Staples (PG KO PEP WMT),
Industrials (CAT BA GE UPS), Utilities (DUK SO NEE).

**Data.** Adjusted closes from Yahoo Finance via `yfinance`. Yahoo revises
adjusted prices over time, so the downloaded CSVs are committed for
reproducibility. `meta.json` records when each dataset was downloaded.

## Notes on the data

These came up in the data review and matter for interpreting the results:
- **Returns are heavy-tailed.** Excess kurtosis is 3.7–26 across stocks for
  daily returns and 2.1–53 for weekly returns (the weekly sample includes
  2008). Jarque–Bera rejects normality for every stock. The glasso
  likelihood is Gaussian, so a zero in Θ̂ means zero *partial correlation*,
  not necessarily conditional independence.
- **A market factor dominates.** The top eigenvalue explains about 38% of the
  variance. It spreads small cross-sector edges across Θ̂. Removing it (or SPY)
  before fitting is a planned comparison.
- **Correlation changes over time.** Average pairwise correlation ranges from
  about 0.07 to 0.8, with peaks in 2008, 2011, 2020 and 2022. This motivates
  fitting glasso separately in high- and low-correlation periods.

## Reference

J. Friedman, T. Hastie, R. Tibshirani. *Sparse inverse covariance estimation
with the graphical lasso.* Biostatistics 9(3), 2008.
