"""
S1: Initial Attempt at Pf2 on the lupus data
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: test Pf2 on lupus data, get visualizations for factor matrices

# load functions/modules ----
from .common import subplotLabel, getSetup, plotFactors, plotWeight
from ..parafac2 import parafac2_nd
from ..imports.scRNA import load_lupus_data
from .common import subplotLabel, getSetup


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (2, 2))  # fig size  # grid size

    # Add subplot labels
    subplotLabel(ax)

    rank = 30

    (
        lupus_tensor,
        _,
        row_colors,
    ) = load_lupus_data()  # don't need to grab cell types here

    weights, factors, _, _ = parafac2_nd(
        lupus_tensor, rank=rank, n_iter_max=20, random_state=1, verbose=True
    )

    plotFactors(
        factors, lupus_tensor, ax[0:3], reorder=(0, 2), trim=(2,), row_colors=row_colors
    )

    plotWeight(weights, ax[3])
    ax[3].set_title("Weight of Each Componenet")

    return f
