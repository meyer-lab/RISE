"""
Lupus: Plot average AUC ROC curve for logistic regression
"""
import numpy as np
import anndata
from .common import subplotLabel, getSetup
from ..logisticReg import getPf2ROC
from sklearn.metrics import RocCurveDisplay


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 6), (1, 1))  # fig size  # grid size

    # Add subplot labels
    subplotLabel(ax)

    X = anndata.read_h5ad(f"/opt/pf2/Lupus_analyzed_40comps.h5ad", backed="r")

    condStatus = X.obs[
        ["Condition", "SLE_status", "Processing_Cohort"]
    ].drop_duplicates()

    y_test, sle_decisions = getPf2ROC(np.array(X.uns["Pf2_A"]), condStatus)

    RocCurveDisplay.from_predictions(
        y_test, sle_decisions, pos_label="SLE", plot_chance_level=True, ax=ax[0]
    )

    ax[0].set_title("OOS ROC: " + str(X.uns["Pf2_A"].shape[1]) + " Component LASSO")

    return f
