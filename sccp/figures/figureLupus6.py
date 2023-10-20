"""
Lupus: Plot average AUC ROC curve for logistic regression
"""
from .common import subplotLabel, getSetup, openPf2
from ..logisticReg import getPf2ROC
from sklearn.metrics import RocCurveDisplay


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 6), (1, 1))  # fig size  # grid size

    # Add subplot labels
    subplotLabel(ax)

    rank = 40
    X = openPf2(rank, "Lupus")
    condStatus = X.obs[
        ["Condition", "SLE_status", "Processing_Cohort", "patient"]
    ].drop_duplicates()
    condStatus = condStatus.set_index("Condition")

    penalties_to_test = [50]
    y_test, sle_decisions = getPf2ROC(
        X.uns["Pf2_A"], condStatus, rank, penalties_to_test=penalties_to_test
    )

    RocCurveDisplay.from_predictions(
        y_test, sle_decisions, pos_label="SLE", plot_chance_level=True, ax=ax[0]
    )

    ax[0].set_title("OOS ROC: " + str(rank) + " Component LASSO")

    return f
