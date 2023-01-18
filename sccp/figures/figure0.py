"""
Creating synthetic data and implementation of parafac2
"""
import numpy as np
from .common import subplotLabel, getSetup, plotSCCP_factors
from ..synthetic import synthXA, plot_synth_pic
from ..parafac2 import parafac2_nd
import pandas as pd


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (3, 4))

    # Add subplot labels
    subplotLabel(ax)

    blobXA, blobDF, celltypeXA = synthXA(magnitude=200 , type="beach")

    weight, factors, projs = parafac2_nd(
        blobXA.to_numpy(),
        rank=2,
    )
    
    plotSCCP_factors(factors, blobXA, celltypeXA, projs[0:2], ax, color_palette)
    
    for i in np.arange(0, 3):
        plot_synth_pic(blobDF, t=i * 3, palette=palette, ax=ax[i + 7])

    return f




palette = {"Ground": "khaki", "Leaf1": "limegreen", "Leaf2": "darkgreen", "Sun": "yellow", "Trunk1": "sienna", "Trunk2": "chocolate"}
color_palette = ["khaki", "limegreen", "darkgreen", "yellow", "sienna", "chocolate"]
