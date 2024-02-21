"""
Lupus: Plot AUC ROC curve for logistic regression for each batch
"""
from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotLupus import plotROCAcrossGroups, getSamplesObs
from ..factorization import correct_conditions


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 6), (1, 1))  # fig size  # grid size

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted.h5ad")
    X.uns["Pf2_A"] = correct_conditions(X)

    plotROCAcrossGroups(
        X.uns["Pf2_A"],
        getSamplesObs(X.obs),
        ax[0],
        pred_group="SLE_status",
        cv_group="Processing_Cohort",
    )

    return f
