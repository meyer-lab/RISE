"""
Creating synthetic data and running ULTRA to calculate factors and recapitulated beach scene data
"""
import numpy as np
from .common import subplotLabel, getSetup, add_ellipse, plotCellAbundance
from gmm.tensor import optimal_seed
from .commonsynthetic import (make_synth_pic, plot_synth_pic, make_blob_tensor, scatterRecapitulated, 
                              plotFactors_synthetic, plotCovFactors_synthetic)

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (5, 4))

    # Add subplot labels
    subplotLabel(ax)
    blob_DF = make_synth_pic(magnitude=100, type="beach")
    
    for i in np.arange(0, 3):
        plot_synth_pic(blob_DF, t=i*3, palette=palette, ax=ax[i])

    rank = 3; n_cluster = 6
    blobXA = make_blob_tensor(blob_DF)

    _, _, fit = optimal_seed(30, blobXA, rank=rank, n_cluster=n_cluster)
    fac = fit[0]
    
    
    plotCellAbundance(fac, n_cluster, ax[3])

    facXA = fac.get_factors_xarray(blobXA)
    DimCol = [f"Dimension{i}" for i in np.arange(1, len(facXA) + 1)]

    scatterRecapitulated(fac, n_cluster, ax)
    plotFactors_synthetic(facXA, DimCol, n_cluster, ax)
    plotCovFactors_synthetic(fac, blobXA, DimCol, n_cluster, ax)

    return f

palette = {"Ground": "khaki", "Trunk": "sienna", "Leaf": "limegreen", "Sun": "yellow"}
