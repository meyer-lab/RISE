"""Factors from scib paper"""
import numpy as np
from .common import subplotLabel, getSetup, plotFactors, plotProj, plotR2X, plotCV, flattenData
from ..imports.scib import import_scib_data
from ..parafac2 import parafac2_nd
import umap.plot
import umap

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (3, 2))

    # Add subplot labels
    subplotLabel(ax)
    
    # data, celltypes = import_scib_data(dataname="ImmuneHuman")
    # data, celltypes = import_scib_data(dataname="ImmuneHumanMouse")
    # data, celltypes = import_scib_data(dataname="Stimulation1")
    data, celltypes = import_scib_data(dataname="Stimulation2")
    # data, celltypes = import_scib_data(dataname="Pancreas")
    
    
    print(celltypes)
    print(len(celltypes))




    # a
    rank = 15

    _, factors, projs, _ = parafac2_nd(
        data,
        rank=rank,
        random_state=1,
    )
    
    dataDF, projDF, _ = flattenData(data, factors, projs)
    
     # UMAP dimension reduction
    pf2Points = umap.UMAP(random_state=1).fit(np.concatenate(projs, axis=0))

    subset = np.random.choice(a=[False, True], size=len(celltypes), p=[.93, .07])
    # for i, drugz in enumerate(celltype):
    #     drugList = np.where(np.asarray(totaldrugs == drugz), drugz, "Other Drugs")
    umap.plot.points(pf2Points, labels=celltypes, ax=ax[3], color_key_cmap="tab20", show_legend=True, subset_points=subset)
    
    umap.plot.points(pf2Points, labels=dataDF["Drug"].values, ax=ax[4], color_key_cmap="tab20", show_legend=True, subset_points=subset)
        # axs[i].set(
        #     title=decomp + "-Based Decomposition",
        # ylabel="UMAP2",
        # xlabel="UMAP1")

    plotFactors(factors, data, ax[0:3], trim=(2,))
    
    return f