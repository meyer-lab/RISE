"""
Creating synthetic data and implementation of parafac2
"""
import numpy as np
import xarray as xa
from .common import (
    subplotLabel,
    getSetup,
    plotFactors,
    plotProj,
    plotSS,
)
from ..synthetic import synthXA, plot_synth_pic
from ..parafac2 import parafac2_nd
from ..decomposition import plotR2X_CC


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    blobInfo, blobDF = synthXA(magnitude=200, type="beach")
    
    # Performing parafac2 on single-cell Xarray
    _, factors, projs, _, _ = parafac2_nd(
        blobInfo["data"].to_numpy(),
        rank=2,
        verbose=True,
    )
    
    plotFactors(factors, blobInfo["data"], ax)

    projs = xa.DataArray(
        projs,
        dims=["Time", "Cell", "Cmp"],
        coords=dict(
            Time=blobInfo.coords["Time"],
            Cell=blobInfo.coords["Cell"],
            Cmp=[f"Cmp. {i}" for i in np.arange(1, projs.shape[2] + 1)]
        ),
        name="projections"
    )
    projs = xa.merge([projs, blobInfo["Cell Type"]], compat="no_conflicts")

    flattened_projs = projs.stack(AllCells=("Time", "Cell"))

    # Remove empty slots
    nonzero_index = np.any(flattened_projs["projections"].to_numpy() != 0, axis=0)
    flattened_projs = flattened_projs.isel(AllCells=nonzero_index)

    plotSS(flattened_projs, ax[3])

    idxx = np.random.choice(len(flattened_projs.coords["AllCells"]), size=200, replace=False)
    plotProj(flattened_projs.isel(AllCells=idxx), ax[4:6])

    plotR2X_CC(blobInfo["data"].to_numpy(), 3, ax[7], ax[8])  

    return f
    
    
palette = {"Ground": "khaki", "Leaf1": "limegreen", "Leaf2": "darkgreen", "Sun": "yellow", "Trunk1": "sienna", "Trunk2": "chocolate"}
color_palette = ["khaki", "limegreen", "darkgreen", "yellow", "sienna", "chocolate"]
