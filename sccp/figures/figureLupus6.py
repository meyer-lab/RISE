"""
Lupus: Plot average AUC ROC curve for logistic regression
"""
import numpy as np
from anndata import read_h5ad
from .common import subplotLabel, getSetup
from ..logisticReg import getPf2ROC
from sklearn.metrics import RocCurveDisplay
from .commonFuncs.plotLupus import getSamplesObs
from ..factorization import correct_conditions


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 6), (1, 1))  # fig size  # grid size

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    condStatus = getSamplesObs(X.obs)
    X.uns["Pf2_A"] = correct_conditions(X)

    y_test, sle_decisions = getPf2ROC(np.array(X.uns["Pf2_A"]), condStatus)

    RocCurveDisplay.from_predictions(
        y_test, sle_decisions, pos_label=True, plot_chance_level=True, ax=ax[0]
    )

    return f
