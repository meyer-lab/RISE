"""
S1: Initial Attempt at Pf2 on the lupus data
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: test Pf2 on lupus data, get visualizations for factor matrices

# load functions/modules ----
from .common import subplotLabel, getSetup, plotFactors, plotWeight, openPf2
from ..imports.scRNA import load_lupus_data


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 16), (2, 2))  # fig size  # grid size

    # Add subplot labels
    subplotLabel(ax)

    rank = 40
    group_to_label = "SLE_status"  # group to label on left side of factor A plot

    (
        lupus_tensor,
        obs,
    ) = load_lupus_data()

    status = obs[["sample_ID", group_to_label]].drop_duplicates("sample_ID")

    # make sure that these two are in the same order
    bool = status["sample_ID"] == lupus_tensor.condition_labels
    assert bool.mean() == 1.0

    group_labs = status.set_index("sample_ID")[group_to_label]

    weights, factors, _ = openPf2(rank=rank, dataName="lupus", optProjs=True)

    plotFactors(
        factors,
        lupus_tensor,
        ax[0:3],
        reorder=(0, 2),
        trim=(2,),
        cond_group_labels=group_labs,
        saveGenes=True,
    )

    plotWeight(weights, ax[3])
    ax[3].set_title("Weight of Each Component")

    return f
