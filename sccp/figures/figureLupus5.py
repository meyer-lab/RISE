"""
Lupus: Plot logistic regression weights for SLE and/or ancestry
"""
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotLupus import plotCmpRegContributions, plot2CmpRegContributions
from ..logisticReg import getCompContribs
import numpy as np
import pandas as pd


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((5, 8), (2, 1))  #

    # Add subplot labels
    subplotLabel(ax)

    rank = 40
    predict = "SLE_status"
    X = openPf2(rank, "Lupus")
    condStatus = X.obs[["Condition", predict]].drop_duplicates()
    condStatus = condStatus.set_index("Condition")
    contribsStatus = getCompContribs(
        X.uns["Pf2_A"], condStatus.to_numpy(), penalty_amt=50
    )
    plotCmpRegContributions(contribsStatus, predict, ax[0])

    predict = "ancestry"
    condStatus = X.obs[["Condition", predict]].drop_duplicates()
    condStatus = condStatus.set_index("Condition")
    condStatus[predict] = np.where(
        condStatus[predict].isin(["European"]), condStatus[predict], "Other"
    )
    contribsAnc = getCompContribs(
        X.uns["Pf2_A"], condStatus[predict].to_numpy(), penalty_amt=50
    )

    contribsStatus["Predicting"] = np.repeat("SLE Status", contribsStatus.shape[0])
    contribsAnc["Predicting"] = np.repeat("Euro-Ancestry", contribsAnc.shape[0])
    plot2CmpRegContributions(pd.concat([contribsStatus, contribsAnc]), ax[1])

    return f
