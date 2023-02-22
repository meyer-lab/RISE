"""
Creating synthetic data and implementation of parafac2
"""
import numpy as np
from .common import subplotLabel, getSetup, plotFactors, plotProjs_SS, renamePlotSynthetic
from ..synthetic import synthXA, plot_synth_pic
from ..parafac2 import parafac2_nd
from ..decomposition import plotR2X_CC


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    blobXA, blobDF, celltypeXA = synthXA(magnitude=200, type="movingcovariance")

    rank = 2
    _, factors, projs, _, _  = parafac2_nd(
        blobXA.to_numpy(),
        rank=rank, verbose=True
    )
    
    plotFactors(factors, blobXA, ax)
    plotProjs_SS(factors, projs[0:9:6], celltypeXA, color_palette, ax)
    
    for i in np.arange(0, 2):
        plot_synth_pic(blobDF[["X","Y","Time","Cell Type"]], t=i*6, palette=palette, type="movingcovariance", ax=ax[i+8])
    
    plotR2X_CC(blobXA.to_numpy(), rank, ax[10], ax[11])
    renamePlotSynthetic(blobXA, ax)


    return f

palette = {"Planet1": "red", "Planet2": "green", "Planet3": "blue", "Planet4": "magenta", "Planet5": "orange"}
color_palette = ["red", "green","blue", "magenta", "orange"]

