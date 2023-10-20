"""
Lupus: Cross validation for determining optimal paramaters for logistic regression
"""
from ..imports.scRNA import import_lupus
from ..logisticReg import testPf2Ranks
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotLupus import plotPf2RankTest


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (1, 1))  # fig size  # grid size

    # Add subplot labels
    subplotLabel(ax)

    X = import_lupus()
    condStatus = X.obs[
        ["Condition", "SLE_status", "Processing_Cohort"]
    ].drop_duplicates()
    condStatus = condStatus.set_index("Condition")

    rank = [2, 3]
    results = testPf2Ranks(
        X,
        condStatus,
        rank,
        cv_group="Processing_Cohort",
    )

    plotPf2RankTest(results, ax[0])

    return f
