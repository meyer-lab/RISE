"""
Lupus: Plot logistic regression weights for SLE and/or ancestry
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotLupus import plotCmpRegContributions
from ..logisticReg import getCompContribs


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
    contribs = pd.concat([contribsStatus, contribsAnc])

    sns.barplot(
        data=contribs,
        x="Component",
        y="Weight",
        hue="Predicting",
        errorbar=None,
        ax=ax[1],
    )
    ax[1].tick_params(axis="x", rotation=90)
    ax[1].set_title("Weight of Pf2 Cmps in Logsitic Regression")

    return f
