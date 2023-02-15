"""
Creating synthetic data and implementation of parafac2
"""
import numpy as np
from .common import subplotLabel, getSetup, plotSCCP_factors, renamePlotSynthetic
from ..synthetic import synthXA, plot_synth_pic
from ..parafac2 import parafac2_nd
from ..decomposition import plotR2X_CC


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    blobXA, blobDF, celltypeXA = synthXA(magnitude=200, type="dividingclusters")

    rank = 2
    _, factors, projs, _, _  = parafac2_nd(
        blobXA.to_numpy(),
        rank=rank, verbose=True
    )
    
    plotSCCP_factors(factors, blobXA, projs[0:9:5], ax, celltypeXA, color_palette, plot_celltype=True)
    
    for i in np.arange(0, 3):
        plot_synth_pic(blobDF[["X","Y","Time","Cell Type"]], t=i * 3, palette=palette, type="dividingclusters", ax=ax[i + 7])
    
    plotR2X_CC(blobXA.to_numpy(), rank, ax[10], ax[11])
    renamePlotSynthetic(blobXA, ax)


    return f



palette = {"Planet1": "lightcoral", "Planet2": "gray", "Planet3": "darkgoldenrod", 
           "Planet4": "magenta", "Planet5": "teal", "Planet6": "deeppink"}

color_palette = ["lightcoral", "gray", "darkgoldenrod", "magenta", "teal", "deeppink"]