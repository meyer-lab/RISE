"""
Figure S5
"""

import seaborn as sns

from scrise.plotting import cell_count_perc_df, rotate_xaxis

from ..imports import import_thomson_factors
from .common import getSetup, subplotLabel


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((5, 5), (1, 1))
    subplotLabel(ax)

    X = import_thomson_factors()

    df = cell_count_perc_df(X, celltype="Cell Type2")
    sns.swarmplot(
        data=df,
        x="Cell Type",
        y="Cell Count",
        color="k",
        ax=ax[0],
    )
    rotate_xaxis(ax[0])
    ax[0].set_yscale("log")

    return f
