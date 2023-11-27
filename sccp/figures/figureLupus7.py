"""
Lupus: Plot AUC ROC curve for logistic regression for each batch
"""
import anndata
from .common import subplotLabel, getSetup
from .commonFuncs.plotLupus import plotROCAcrossGroups


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 6), (1, 1))  # fig size  # grid size

    # Add subplot labels
    subplotLabel(ax)

    X = anndata.read_h5ad(f"/opt/pf2/Lupus_analyzed_40comps.h5ad", backed="r")

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
