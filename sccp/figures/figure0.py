"""
Creating synthetic data and implementation of parafac2
"""
import numpy as np
from .common import subplotLabel, getSetup, plotSCCP_factors
from ..synthetic import synthXA, plot_synth_pic
from ..parafac2 import parafac2


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    blobXA, blobDF = synthXA(magnitude=200, type="beach")

    _, factors, projs = parafac2(
        blobXA.to_numpy(),
        rank=3,
        verbose=True,
    )
    
    plotSCCP_factors(factors, blobXA, projs[0:2], ax)

    for i in np.arange(0, 3):
        plot_synth_pic(blobDF, t=i * 3, palette=palette, ax=ax[i + 5])

    return f


palette = {"Ground": "khaki", "Trunk": "sienna", "Leaf": "limegreen", "Sun": "yellow"}
