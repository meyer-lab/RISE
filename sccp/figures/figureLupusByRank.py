"""
Lupus: Cross validation for determining optimal paramaters for logistic regression
Lupus is spelled wrong in the figure name to indicate that this figure does not depend
on cached results.
"""
import numpy as np
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

    condStatus = getSamplesObs(X.obs)

    rank = np.arange(5, 55, 5)
    results = testPf2Ranks(
        X,
        condStatus,
        rank,
        cv_group="Processing_Cohort",
    )

    plotPf2RankTest(results, ax[0])

    return f
