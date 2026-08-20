"""
Plotting functions for BiCV-based rank selection.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from ..rank_selection import BiCVOptimizationResult


def plot_bicv_r2x(results: pd.DataFrame, ax: Axes) -> None:
    """Plot BiCV R2X and in-sample fit R2X across ranks.

    The fit R2X (in-sample, computed on the full dataset) increases
    monotonically with rank. The BiCV R2X (held-out, averaged across
    repeated random cell/gene splits) penalizes overfitting and typically
    peaks near the rank that best generalizes to unseen data. The peak (or
    plateau) of the BiCV R2X curve is a good candidate for the rank to use.

    Parameters
    ----------
    results : pandas.DataFrame
        Output of :func:`RISE.rank_selection.bicv`, with columns "Rank",
        "Repeat", "Metric" ("Fit R2X" or "BiCV R2X"), and "R2X".
    ax : matplotlib.axes.Axes
        Matplotlib axes object to plot on.
    """
    sns.lineplot(
        data=results,
        x="Rank",
        y="R2X",
        hue="Metric",
        style="Metric",
        markers=True,
        dashes=False,
        errorbar="sd",
        ax=ax,
    )
    ax.set(xlabel="Rank", ylabel="R2X")
    ax.legend(title=None)


def plot_rank_optimization(result: BiCVOptimizationResult, ax: Axes) -> None:
    """Plot the Bayesian optimization trace used to select a rank.

    Shows the Gaussian process surrogate's posterior mean and 95%
    confidence interval over the searched rank range, the individual ranks
    that were actually evaluated (mean BiCV R2X across repeats), and the
    best rank found.

    Parameters
    ----------
    result : RISE.rank_selection.BiCVOptimizationResult
        Output of :func:`RISE.rank_selection.optimize_rank`.
    ax : matplotlib.axes.Axes
        Matplotlib axes object to plot on.
    """
    lo, hi = result.rank_bounds
    grid = np.arange(lo, hi + 1)
    mu, sigma = result.gp.predict(grid.reshape(-1, 1).astype(float), return_std=True)

    ax.plot(grid, mu, color="black", label="GP mean")
    ax.fill_between(
        grid,
        mu - 1.96 * sigma,
        mu + 1.96 * sigma,
        color="black",
        alpha=0.15,
        label="95% CI",
    )
    ax.errorbar(
        result.history["Rank"],
        result.history["R2X"],
        yerr=result.history["R2X_std"],
        fmt="o",
        color="tab:blue",
        zorder=3,
        label="Evaluated ranks",
    )
    ax.axvline(
        result.best_rank,
        color="tab:red",
        linestyle="--",
        label=f"Best rank ({result.best_rank})",
    )
    ax.set(xlabel="Rank", ylabel="BiCV R2X")
    ax.legend()
