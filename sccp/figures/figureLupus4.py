"""
S3: Logisitic Regression (and maybe SVM) on Pf2 Factor matrix A output
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: Test various Pf2 ranks to see which best predicts disease status

# load functions/modules ----
from .common import subplotLabel, getSetup, plotPf2RankTest
from ..imports.scRNA import load_lupus_data
from ..logisticReg import testPf2Ranks


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (1, 1))  # fig size  # grid size

    # Add subplot labels
    subplotLabel(ax)

    lupus_tensor, obs = load_lupus_data()

    status = obs[["sample_ID", "SLE_status", "Processing_Cohort"]].drop_duplicates()

    group_labs = status.set_index("sample_ID")[["SLE_status", "Processing_Cohort"]]

    ranks_to_test = [2, 3]  # set to 2,3 for github test
    penalties_to_test = [10, 25, 50, 100, 200]

    results = testPf2Ranks(
        lupus_tensor,
        group_labs,
        ranks_to_test,
        penalties_to_test=penalties_to_test,
        cv_group="Processing_Cohort",
    )
    print(results)
    plotPf2RankTest(results, ax[0])

    return f
