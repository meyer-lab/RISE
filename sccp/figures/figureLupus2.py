"""
S2: Examining cell state for Pf2 on lupus data
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: visualize the cell state compostition by cell type/UMAP

from .common import (
    subplotLabel,
    getSetup,
    plotCmpUMAP,
    plotUMAP_obslabel,
    openPf2,
    openUMAP,
)
from ..imports.scRNA import load_lupus_data


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 13), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    rank = 40
    cmp = 13

    # Import of data
    _, obs = load_lupus_data()

    broad_type = obs["cell_type_broad"].reset_index(drop=True)
    lympho_type = obs["cell_type_lympho"].reset_index(drop=True)

    # replace NaN with string
    lympho_type = lympho_type.cat.add_categories("other").fillna("other")

    (
        _,
        factors,
        projs,
    ) = openPf2(rank=rank, dataName="lupus", optProjs=True)

    # UMAP dimension reduction
    pf2Points = openUMAP(40, "lupus", opt=True)

    plotCmpUMAP(cmp, factors, pf2Points, projs, ax[0])
    plotUMAP_obslabel(broad_type, pf2Points, ax[1])
    plotUMAP_obslabel(lympho_type, pf2Points, ax[2])

    return f
