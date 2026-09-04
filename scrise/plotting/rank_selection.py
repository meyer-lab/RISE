"""
Plotting functions for BiCV-based rank selection.
"""

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


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
        Output of :func:`scrise.rank_selection.bicv`, with columns "Rank",
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
