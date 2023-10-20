"""
Lupus: Plot AUC ROC curve for logistic regression for each batch
"""
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

    plotROCAcrossGroups(
        X.uns["Pf2_A"],
        condStatus,
        ax[0],
        pred_group=predict,
        cv_group="Processing_Cohort",
    )

    return f
