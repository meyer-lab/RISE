"""
S3: Logisitic Regression (and maybe SVM) on Pf2 Factor matrix A output
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: Test various Pf2 ranks to see which best predicts disease status

# load functions/modules ----
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
