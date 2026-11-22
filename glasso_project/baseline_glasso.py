"""Baseline: scikit-learn graphical lasso on the raw returns.

Run from the PROJECT folder (the one containing glasso_project/).

Reference point for our own implementation. Deliberately minimal preprocessing:
returns are only z-scored (so the input S is the sample correlation matrix and the
penalty alpha is on a unit-free scale). No factor removal, no rank transform,
no winsorizing, full sample.

Note on the penalty: sklearn penalizes only off-diagonal entries of Theta
(alpha * sum_{i!=j} |Theta_ij|). Friedman et al.'s original glasso (and our
glasso.py, which starts from W = S + rho*I) also penalizes the diagonal, so the
two give slightly different Theta for the same alpha.

Reads  data/<name>/returns.csv
Writes data/<name>/glasso_sklearn/{precision,partial_corr,edges,path}.csv,
       summary.json, plots/*.png

    python -m glasso_project.baseline_glasso 1wk_2005_2026          # alpha chosen by 5-fold CV
    python -m glasso_project.baseline_glasso 1d_2019_2026 --alpha 0.1
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV, graphical_lasso

from glasso_project.data_pipeline.config import SECTORS, dataset_dir, list_datasets, load_meta
from glasso_project.network_plot import LAYOUTS, draw_network
from glasso_project.data_pipeline.plot import sector_order_and_colors

SECTOR_OF = {t: s for s, ts in SECTORS.items() for t in ts}
# sklearn's default inner-lasso tolerance (1e-4) stalls short of the optimum at some
# alphas (ConvergenceWarning, KKT violations ~3e-3); 1e-7 converges everywhere.
ENET_TOL = 1e-7


def kkt_violation(S, W, theta, alpha, tol=1e-10):
    """Max violation of the off-diagonal optimality conditions:
    W_ij - S_ij = alpha*sign(Theta_ij) if Theta_ij != 0, |W_ij - S_ij| <= alpha otherwise."""
    G = W - S
    off = ~np.eye(len(S), dtype=bool)
    nz = (np.abs(theta) > tol) & off
    z = ~nz & off
    v1 = np.abs(G[nz] - alpha * np.sign(theta[nz])).max() if nz.any() else 0.0
    v2 = max(0.0, (np.abs(G[z]) - alpha).max()) if z.any() else 0.0
    return float(max(v1, v2))


def partial_corr(theta):
    d = np.sqrt(np.diag(theta))
    P = -theta / np.outer(d, d)
    np.fill_diagonal(P, 1.0)
    return P


def edge_list(P, tickers, tol=1e-8):
    i, j = np.triu_indices_from(P, 1)
    m = np.abs(P[i, j]) > tol
    e = pd.DataFrame({"a": np.array(tickers)[i[m]], "b": np.array(tickers)[j[m]],
                      "partial_corr": P[i[m], j[m]]})
    e["same_sector"] = [SECTOR_OF.get(a) == SECTOR_OF.get(b) for a, b in zip(e.a, e.b)]
    return e.reindex(e.partial_corr.abs().sort_values(ascending=False).index)


def fit(name, alpha=None, n_alphas=30, cv=5, min_abs=0.0):
    meta = load_meta(name)
    d = dataset_dir(name)
    out = d / "glasso_sklearn"
    (out / "plots").mkdir(parents=True, exist_ok=True)

    r = pd.read_csv(d / "returns.csv", index_col=0, parse_dates=True)
    order, _, _ = sector_order_and_colors(list(r.columns))
    X = r[order].values
    X = (X - X.mean(0)) / X.std(0)
    n, p = X.shape
    S = np.corrcoef(X, rowvar=False)

    # alpha >= max off-diagonal |S_ij| gives the empty graph; grid below that
    a_max = np.abs(S - np.eye(p)).max()
    alphas = np.logspace(np.log10(a_max * 1e-3), np.log10(a_max), n_alphas)

    # KFold without shuffling -> contiguous time blocks as folds
    cvm = GraphicalLassoCV(alphas=list(alphas), cv=cv, max_iter=500, enet_tol=ENET_TOL).fit(X)
    cv_alpha = cvm.alpha_
    use_alpha = alpha if alpha is not None else cv_alpha

    # regularization path: edges / within-sector share at each alpha
    rows = []
    for a in alphas:
        _, th = graphical_lasso(S, alpha=a, max_iter=500, enet_tol=ENET_TOL)
        e = edge_list(partial_corr(th), order)
        rows.append(dict(alpha=a, n_edges=len(e),
                         within_sector_frac=e.same_sector.mean() if len(e) else np.nan))
    path = pd.DataFrame(rows)
    cv_a = np.asarray(cvm.cv_results_["alphas"])
    cv_s = np.asarray(cvm.cv_results_["mean_test_score"])
    path["cv_loglik"] = [cv_s[np.argmin(np.abs(cv_a - a))] for a in alphas]
    path.to_csv(out / "path.csv", index=False)

    cov, theta = graphical_lasso(S, alpha=use_alpha, max_iter=500, enet_tol=ENET_TOL)
    P = partial_corr(theta)
    edges = edge_list(P, order)
    pd.DataFrame(theta, order, order).to_csv(out / "precision.csv")
    pd.DataFrame(P, order, order).to_csv(out / "partial_corr.csv")
    edges.to_csv(out / "edges.csv", index=False)

    n_pairs = p * (p - 1) // 2
    within_pairs = sum(len([t for t in ts if t in order]) * (len([t for t in ts if t in order]) - 1) // 2
                       for ts in SECTORS.values())
    summary = dict(
        dataset=name, interval=meta["interval"], n=n, p=p,
        alpha_cv=float(cv_alpha), alpha_used=float(use_alpha), alpha_max=float(a_max),
        n_edges=len(edges), density=len(edges) / n_pairs,
        within_sector_frac=float(edges.same_sector.mean()) if len(edges) else None,
        within_sector_frac_if_random=within_pairs / n_pairs,
        n_negative_partial_corr=int((edges.partial_corr < 0).sum()),
        kkt_violation=kkt_violation(S, cov, theta, use_alpha),
        max_abs_W_theta_minus_I=float(np.abs(cov @ theta - np.eye(p)).max()),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    _plot_path(path, cv_alpha, use_alpha, summary, out / "plots" / "01_path.png")
    _plot_heatmaps(S, P, order, use_alpha, out / "plots" / "02_corr_vs_partial.png")
    old = out / "plots" / "03_network.png"  # pre-fixed-layout version
    if old.exists():
        old.unlink()
    _plot_networks(edges, order, use_alpha, name, out / "plots", min_abs=min_abs)

    print(f"[{name}] n={n}, p={p}; CV alpha={cv_alpha:.4f} (alpha_max={a_max:.3f}); "
          f"using alpha={use_alpha:.4f}")
    print(f"  edges: {len(edges)}/{n_pairs} (density {summary['density']:.1%}), "
          f"within-sector {summary['within_sector_frac']:.1%} "
          f"(random graph: {summary['within_sector_frac_if_random']:.1%}), "
          f"negative partial corr: {summary['n_negative_partial_corr']}")
    print("  strongest edges:")
    print(edges.head(10).to_string(index=False, float_format="{:.3f}".format))
    print(f"  outputs in {out}")
    return summary


def _plot_path(path, cv_alpha, use_alpha, s, fname):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    ax = axes[0]
    ax.semilogx(path.alpha, path.cv_loglik, "o-")
    ax.axvline(cv_alpha, color="r", ls="--", label=f"CV alpha={cv_alpha:.3g}")
    if use_alpha != cv_alpha:
        ax.axvline(use_alpha, color="k", ls=":", label=f"used alpha={use_alpha:.3g}")
    ax.set_xlabel("alpha"); ax.set_ylabel("mean held-out log-likelihood")
    ax.set_title("Cross-validation"); ax.legend()
    ax = axes[1]
    ax.semilogx(path.alpha, path.n_edges, "o-", label="# edges")
    ax.axvline(use_alpha, color="r", ls="--")
    ax.set_xlabel("alpha"); ax.set_ylabel("# edges"); ax.set_title("Regularization path")
    ax2 = ax.twinx()
    ax2.semilogx(path.alpha, path.within_sector_frac, "s-", color="C1",
                 label="within-sector share")
    ax2.axhline(s["within_sector_frac_if_random"], color="C1", ls=":",
                label="share if edges random")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("within-sector share of edges")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center left", fontsize=8)
    fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)


def _plot_heatmaps(S, P, order, alpha, fname):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    Pd = P.copy(); np.fill_diagonal(Pd, np.nan)
    Sd = S.copy(); np.fill_diagonal(Sd, np.nan)
    for ax, M, title in [(axes[0], Sd, "Sample correlation"),
                         (axes[1], Pd, f"Glasso partial correlation (alpha={alpha:.3g})")]:
        lim = np.nanmax(np.abs(M))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=7)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7)
        ax.set_title(title + " (diagonal blank)")
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)


def _plot_networks(edges, order, alpha, name, out_dir, layouts=LAYOUTS, min_abs=0.0):
    for lay in layouts:
        draw_network(edges, tickers=order, layout=lay, min_abs=min_abs,
                     title=f"sklearn glasso — {name}, alpha={alpha:.3g}",
                     fname=out_dir / f"03_network_{lay}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help=f"dataset name; available: {list_datasets()}")
    ap.add_argument("--alpha", type=float, help="fixed penalty (default: chosen by CV)")
    ap.add_argument("--cv", type=int, default=5, help="number of contiguous CV folds")
    ap.add_argument("--show-min", type=float, default=0.0,
                    help="network plot: hide edges with |partial corr| below this "
                         "(display only; all edges are still saved)")
    a = ap.parse_args()
    fit(a.name, a.alpha, cv=a.cv, min_abs=a.show_min)


if __name__ == "__main__":
    main()
