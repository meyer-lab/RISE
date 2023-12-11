"""
Lupus: Cross validation for determining optimal paramaters for logistic regression
"""
from ..imports import import_lupus
from ..logisticReg import testPf2Ranks
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotLupus import plotPf2RankTest
from .commonFuncs.plotLupus import getSamplesObs


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (1, 1))  # fig size  # grid size

    # Add subplot labels
    subplotLabel(ax)

    X = import_lupus()
    X = X.to_memory()

    condStatus = lupusStatus = getSamplesObs(X.obs)

    rank = [2, 3]
    # results = testPf2Ranks(
    #     X,
    #     condStatus,
    #     rank,
    #     cv_group="Processing_Cohort",
    # )

    # plotPf2RankTest(results, ax[0])

    return f
