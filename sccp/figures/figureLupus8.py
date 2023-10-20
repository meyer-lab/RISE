"""
S4: Investigation of Component 13
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: look at the cells that are highly contributing to component 13; see if they're spread evenly
# note: they are spread evenly among cell types, BUT there is a specific original louvain cluster (14) that
# is almost exclusively them-- likely OG megakaryocyte cluster that was removed/clumped in with others

# load functions/modules ----
from .common import subplotLabel, getSetup, openPf2
from .commonFuncs.plotLupus import investigate_comp


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 8), (3, 2))  # fig size  # grid size

    # Add subplot labels
    subplotLabel(ax)

    rank = 40
    component = 13

    X = openPf2(rank=40, dataName="Lupus")

    investigate_comp(
        X, component, "Cell Type", ax[0], threshold=0.1
    )

    return f
