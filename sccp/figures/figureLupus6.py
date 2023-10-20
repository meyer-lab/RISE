"""
S3: Logisitic Regression (and maybe SVM) on Pf2 Factor matrix A output
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: run logisitc regression and assess predictive ability of 40 comp Pf2 using ROC AUC
# comparison to Perez et al (linked above) fig 4C

# load functions/modules ----
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
    condStatus = X.obs[["Condition", "SLE_status", "Processing_Cohort", "patient"]].drop_duplicates()
    condStatus = condStatus.set_index("Condition")

    penalties_to_test = [50]
    # get test data, and decisions made by the trained model corresponding to those test data
    y_test, sle_decisions = getPf2ROC(X.uns["Pf2_A"], condStatus, rank, penalties_to_test=penalties_to_test)

    # make plot of ROC AUC
    RocCurveDisplay.from_predictions(
        y_test, sle_decisions, pos_label="SLE", plot_chance_level=True, ax=ax[0]
    )

    ax[0].set_title("OOS ROC: " + str(rank) + " Component LASSO")

    return f
