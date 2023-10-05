"""
S3b: Logistic Regression (and maybe SVM) on Pf2 Factor matrix A output
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: run logisitc regression to see which components are best able to predict disease status

# load functions/modules ----
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotLupus import plotCmpRegContributions, plot2CmpRegContributions
from ..imports.scRNA import load_lupus_data
from ..logisticReg import getCompContribs
import numpy as np
import pandas as pd

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((5, 8), (2, 1))  # fig size  # grid size

    # Add subplot labels
    subplotLabel(ax)

    rank = 40
    group_to_predict = "SLE_status"  # group to predict in logistic regression
    _, obs = load_lupus_data()
    status = obs[["sample_ID", group_to_predict]].drop_duplicates()
    group_labs = status.set_index("sample_ID")[group_to_predict]

    _, factors, _ = openPf2(rank=rank, dataName="lupus", optProjs=True)

    A_matrix = factors[0]
    contribsStatus = getCompContribs(A_matrix, group_labs.to_numpy(), penalty_amt=50)
    plotCmpRegContributions(contribsStatus, group_to_predict, ax[0])

    contribsStatus["Predicting"] = np.repeat("SLE Status", contribsStatus.shape[0])

    group_to_predict = "ancestry"  # group to predict in logistic regression
    status = obs[["sample_ID", group_to_predict]].drop_duplicates()
    group_labs = status.set_index("sample_ID")
    group_labs[group_to_predict] = np.where(group_labs[group_to_predict].isin(["European"]), group_labs[group_to_predict], "Other")

    contribsAnc = getCompContribs(A_matrix, group_labs[group_to_predict].to_numpy(), penalty_amt=50)
    contribsAnc["Predicting"] = np.repeat("Euro-Ancestry", contribsAnc.shape[0])

    plot2CmpRegContributions(pd.concat([contribsStatus,contribsAnc]), ax[1])

    return f
