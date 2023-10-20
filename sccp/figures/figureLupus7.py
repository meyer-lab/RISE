"""
S3b: Logistic Regression (and maybe SVM) on Pf2 Factor matrix A output
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: run logisitc regression to see which components are best able to predict disease status

# load functions/modules ----
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotLupus import plotROCAcrossGroups


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 6), (1, 1))  # fig size  # grid size

    # Add subplot labels
    subplotLabel(ax)

    rank = 40
    X = openPf2(rank, "Lupus")
    predict = "SLE_status"
    condStatus = X.obs[["Condition", predict, "Processing_Cohort"]].drop_duplicates()
    condStatus = condStatus.set_index("Condition")
    
    # predict = "ancestry"
    # condStatus["ancestry"] = np.where(condStatus["ancestry"].isin(["European"]), condStatus["ancestry"], "Other")

    plotROCAcrossGroups(
        X.uns["Pf2_A"],
        condStatus,
        ax[0],
        pred_group=predict,
        cv_group="Processing_Cohort",
    )

    return f
