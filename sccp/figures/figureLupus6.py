"""
S3: Logisitic Regression (and maybe SVM) on Pf2 Factor matrix A output
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: run logisitc regression and assess predictive ability of 40 comp Pf2 using ROC AUC
# comparison to Perez et al (linked above) fig 4C

# load functions/modules ----
from .common import subplotLabel, getSetup, openPf2

from ..imports.scRNA import load_lupus_data
from ..logisticReg import getPf2ROC
from sklearn.metrics import RocCurveDisplay


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 6), (1, 1))  # fig size  # grid size

    # Add subplot labels
    subplotLabel(ax)

    rank = 40

    _, obs = load_lupus_data()

    status = obs[
        ["sample_ID", "SLE_status", "Processing_Cohort", "patient"]
    ].drop_duplicates()

    group_labs = status.set_index("sample_ID")

    (
        _,
        factors,
        _,
    ) = openPf2(rank=rank, dataName="lupus", optProjs=True)

    A_matrix = factors[0]

    # only doing 50 because that is the penalty we used when we chose 40 as an optimal component number
    penalties_to_test = [50]

    # get test data, and decisions made by the trained model corresponding to those test data
    y_test, sle_decisions = getPf2ROC(
        A_matrix, group_labs, rank, penalties_to_test=penalties_to_test, 
    )

    # make plot of ROC AUC
    RocCurveDisplay.from_predictions(
        y_test, sle_decisions, pos_label="SLE", plot_chance_level=True, ax=ax[0]
    )

    ax[0].set_title("OOS ROC for Cases/Controls: " + str(rank) + " Component LASSO")

    return f
