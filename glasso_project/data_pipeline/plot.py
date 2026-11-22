"""Step 3: figures from prices/returns/stats.

Run from the PROJECT folder (the one containing glasso_project/).

Reads  data/<name>/{prices,returns,summary_stats}.csv
Writes data/<name>/plots/*.png

    python -m glasso_project.data_pipeline.plot 1wk_2005_2026
    python -m glasso_project.data_pipeline.plot 1d_2019_2026 --only 1 5      # just figures 01 and 05
    python -m glasso_project.data_pipeline.plot 1wk_2005_2026 --window 52    # rolling window in periods
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from glasso_project.data_pipeline.config import DEFAULT_WINDOW, SECTORS, dataset_dir, list_datasets, load_meta


def sector_order_and_colors(tickers):
    """Tickers ordered by sector, plus {ticker: color} and {sector: color}."""
    cmap = plt.get_cmap("tab10")
    sec_color = {s: cmap(i) for i, s in enumerate(SECTORS)}
    sec_color["Other"] = (0.5, 0.5, 0.5, 1.0)
    sector_of = {t: s for s, ts in SECTORS.items() for t in ts}
    order = [t for ts in SECTORS.values() for t in ts if t in tickers]
    order += [t for t in tickers if t not in order]
    colors = {t: sec_color[sector_of.get(t, "Other")] for t in order}
    used = {s: c for s, c in sec_color.items()
            if any(sector_of.get(t, "Other") == s for t in order)}
    return order, colors, used


class Plotter:
    def __init__(self, name, window=None):
        self.meta = load_meta(name)
        self.d = dataset_dir(name)
        missing = [f for f in ("returns.csv", "summary_stats.csv")
                   if not (self.d / f).exists()]
        if missing:
            raise SystemExit(f"{self.d} missing {missing}; run `python -m glasso_project.data_pipeline.analyze {name}` first.")
        self.r = pd.read_csv(self.d / "returns.csv", index_col=0, parse_dates=True)
        self.s = pd.read_csv(self.d / "summary_stats.csv", index_col=0)
        self.prices = pd.read_csv(self.d / "prices.csv", index_col=0,
                                  parse_dates=True)[self.r.columns]
        self.window = window or DEFAULT_WINDOW[self.meta["interval"]]
        self.tag = f"{name} ({self.meta['interval']})"
        self.plots = self.d / "plots"
        self.plots.mkdir(exist_ok=True)

    def _save(self, fig, fname, rect=None):
        fig.tight_layout(rect=rect)
        fig.savefig(self.plots / fname, dpi=150)
        plt.close(fig)

    # 01
    def prices_plot(self):
        order, colors, used = sector_order_and_colors(list(self.prices.columns))
        p = self.prices[order]
        norm = p / p.bfill().iloc[0] * 100
        fig, ax = plt.subplots(figsize=(11, 6))
        for t in order:
            ax.plot(norm.index, norm[t], lw=0.8, color=colors[t], alpha=0.8)
        ax.set_yscale("log")
        ax.set_title(f"Adjusted close, rebased to 100 (log scale) — {self.tag}")
        ax.set_ylabel("Index")
        handles = [plt.Line2D([], [], color=c, lw=2) for c in used.values()]
        ax.legend(handles, used.keys(), title="Sector", loc="upper left", fontsize=8)
        self._save(fig, "01_prices.png")

    # 02
    def returns_ts(self):
        r, w = self.r, self.window
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        r.plot(ax=axes[0], lw=0.4, legend=False, alpha=0.7)
        axes[0].set_title(f"Returns, all tickers — {self.tag}")
        axes[1].plot(r.rolling(w).std().mean(axis=1))
        axes[1].set_title(f"Cross-sectional mean rolling {w}-period volatility")
        self._save(fig, "02_returns_timeseries.png")

    # 03
    def distributions(self):
        r = self.r
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        pooled = ((r - r.mean()) / r.std()).values.ravel()
        axes[0].hist(pooled, bins=120, density=True, alpha=0.6,
                     label="standardized returns")
        x = np.linspace(-6, 6, 400)
        axes[0].plot(x, stats.norm.pdf(x), "r", label="N(0,1)")
        axes[0].set_xlim(-8, 8); axes[0].set_yscale("log"); axes[0].set_ylim(1e-5, 1)
        axes[0].set_title("Pooled standardized returns (log y)"); axes[0].legend()
        stats.probplot(pooled, dist="norm", plot=axes[1])
        axes[1].set_title("Normal Q-Q (pooled)")
        axes[2].hist(self.s["excess_kurt"], bins=15)
        axes[2].set_title("Excess kurtosis across tickers")
        self._save(fig, "03_distributions.png")

    # 04
    def stats_bars(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, col in zip(axes, ["ann_return", "ann_vol", "excess_kurt"]):
            self.s[col].sort_values().plot.barh(ax=ax)
            ax.set_title(col)
        self._save(fig, "04_summary_stats.png")

    # 05
    def correlation(self):
        order, _, _ = sector_order_and_colors(list(self.r.columns))
        C = self.r[order].corr()
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=7)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"Sample correlation (sector-ordered) — {self.tag}")
        self._save(fig, "05_correlation.png")

    # 06
    def spectrum(self, n_vecs=3):
        n, p = self.r.shape
        order, colors, used = sector_order_and_colors(list(self.r.columns))
        ev, V = np.linalg.eigh(self.r[order].corr().values)
        ev, V = ev[::-1], V[:, ::-1]
        V = V * np.sign(V.sum(axis=0))  # fix sign so loadings are mostly positive
        q = p / n  # Marchenko-Pastur bulk edges for correlation matrix
        lo, hi = (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2

        fig = plt.figure(figsize=(14, 9))
        gs = fig.add_gridspec(2, n_vecs)
        ax0 = fig.add_subplot(gs[0, : n_vecs // 2 + 1])
        ax1 = fig.add_subplot(gs[0, n_vecs // 2 + 1:])
        ax0.bar(range(1, p + 1), ev)
        ax0.axhspan(lo, hi, color="r", alpha=0.2, label="Marchenko-Pastur bulk")
        ax0.set_title(f"Correlation eigenvalues — {self.tag}")
        ax0.set_xlabel("index"); ax0.legend()
        ax1.plot(range(1, p + 1), np.cumsum(ev) / ev.sum(), "o-")
        ax1.set_title("Cumulative variance explained"); ax1.set_xlabel("# eigenvalues")
        cols = [colors[t] for t in order]
        for k in range(n_vecs):
            ax = fig.add_subplot(gs[1, k])
            ax.bar(range(p), V[:, k], color=cols)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(range(p)); ax.set_xticklabels(order, rotation=90, fontsize=6)
            ax.set_title(f"Eigenvector {k + 1} loadings "
                         f"(λ={ev[k]:.2f}, {ev[k] / ev.sum():.1%} var)")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in used.values()]
        fig.legend(handles, used.keys(), title="Sector", loc="lower center",
                   ncol=len(used), frameon=False)
        self._save(fig, "06_eigenspectrum.png", rect=(0, 0.05, 1, 1))

    # 07
    def rolling_corr(self):
        r, w = self.r, self.window
        iu = np.triu_indices(r.shape[1], 1)
        avg = r.rolling(w).corr().groupby(level=0).apply(
            lambda m: m.values[iu].mean()).dropna()
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(avg)
        ax.set_title(f"Average pairwise correlation (rolling {w} periods) — {self.tag}")
        self._save(fig, "07_rolling_correlation.png")

    def run(self, only=None):
        figs = [self.prices_plot, self.returns_ts, self.distributions, self.stats_bars,
                self.correlation, self.spectrum, self.rolling_corr]
        for i, f in enumerate(figs, 1):
            if not only or i in only:
                f()
        print(f"Plots in {self.plots}")


def plot(name, window=None, only=None):
    Plotter(name, window).run(only)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help=f"dataset name; available: {list_datasets()}")
    ap.add_argument("--window", type=int, help="rolling window in periods "
                    "(default 60 daily / 26 weekly / 12 monthly)")
    ap.add_argument("--only", type=int, nargs="+", help="figure numbers to make (1-7)")
    a = ap.parse_args()
    plot(a.name, a.window, a.only)


if __name__ == "__main__":
    main()
