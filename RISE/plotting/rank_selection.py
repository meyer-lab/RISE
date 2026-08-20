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
    """Plot the quadratic-guided search trace used to select a rank.

    Shows the least-squares quadratic fit of BiCV R2X vs. rank (used to
    guide which ranks were evaluated), the individual ranks that were
    actually evaluated (mean +/- std BiCV R2X across repeats), and the best
    rank found. The quadratic fit's R² is noted in its legend label; a low
    value indicates the curve is not well described by a quadratic over the
    searched range, and ``result.best_rank`` (rather than the fit's vertex)
    should be trusted.

    Parameters
    ----------
    result : RISE.rank_selection.BiCVOptimizationResult
        Output of :func:`RISE.rank_selection.optimize_rank`.
    ax : matplotlib.axes.Axes
        Matplotlib axes object to plot on.
    """
    lo, hi = result.rank_bounds
    grid = np.arange(lo, hi + 1)
    fitted = result.quadratic_fit(grid)

    ax.plot(
        grid,
        fitted,
        color="black",
        label=f"Quadratic fit (R²={result.quadratic_r2:.2f})",
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
