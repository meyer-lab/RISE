"""
Creating synthetic data and implementation of parafac2
"""
import numpy as np
from .common import subplotLabel, getSetup, plotSCCP_factors
from ..synthetic import synthXA, plot_synth_pic
from ..parafac2 import parafac2_nd
from ..decomposition import plotR2X


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    blobXA, blobDF, celltypeXA = synthXA(magnitude=200 , type="beach")

    rank = 2
    weight, factors, projs = parafac2_nd(
        blobXA.to_numpy(),
        rank=rank, verbose=True
    )
    
    plotSCCP_factors(factors, blobXA, projs[0:2], ax, celltypeXA, color_palette, plot_celltype=True)
    
    for i in np.arange(0, 3):
        plot_synth_pic(blobDF, t=i * 3, palette=palette, type="beach", ax=ax[i + 7])
    
    plotR2X(blobXA.to_numpy(), rank, "Synthetic1", ax[11], run_decomp=False)

    return f

palette = {"Ground": "khaki", "Leaf1": "limegreen", "Leaf2": "darkgreen", "Sun": "yellow", "Trunk1": "sienna", "Trunk2": "chocolate"}
color_palette = ["khaki", "limegreen", "darkgreen", "yellow", "sienna", "chocolate"]