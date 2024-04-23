"""
Lupus: Plot average AUC ROC curve for logistic regression
"""

from anndata import read_h5ad
from .common import subplotLabel, getSetup
from ..logisticReg import roc_lupus_fourtbatch
from sklearn.metrics import RocCurveDisplay
from .commonFuncs.plotLupus import samples_only_lupus
from ..factorization import correct_conditions


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((3, 3), (1, 1)) 

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    X.uns["Pf2_A"] = correct_conditions(X)
    
    plot_roc_fourthbatch(X, ax[0])
    

    return f



def plot_roc_fourthbatch(X, ax):
    """Plots ROC curve for prediction """
    y_test, sle_decisions = roc_lupus_fourtbatch(X, samples_only_lupus(X))

    RocCurveDisplay.from_predictions(
        y_test, sle_decisions, pos_label=True, plot_chance_level=True, ax=ax
    )
