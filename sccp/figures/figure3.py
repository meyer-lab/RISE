"""
Parafac2 implementation on PBMCs treated across IL2 treatments, times, and doses
"""
import numpy as np
from .common import subplotLabel, getSetup, plotSCCP_factors, renamePlotIL2
from ..imports.cytok import IL2_flowXA
from ..parafac2 import parafac2_nd


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Ligand, Dose, Time, Cell, Marker]
    flowXA, celltypeXA = IL2_flowXA(saveXA=False, IncludeNoneCells=False)

    # flowXA -= np.mean(flowXA, axis=(0, 1, 2, 3))
    # flowXA /= np.std(flowXA, axis=(0, 1, 2, 3))


    # Shrink dataset
    flowXA = flowXA.loc[:, :, :, :50, :]
    celltypeXA = celltypeXA.loc[:, :, :, :50]

    # Performing parafac2 on single-cell Xarray
    _, factors, projs = parafac2_nd(flowXA.to_numpy(), rank=3, verbose=True) 

    plotSCCP_factors(factors, flowXA, projs[0, 0, :, :, :], ax, celltypeXA[0, 0, :, :], color_palette, plot_celltype=True)
    renamePlotIL2(ax)
    
    return f

color_palette = ["red", "blue", "green"]