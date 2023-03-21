"""
Parafac2 implementation on PBMCs treated wtih PopAlign/Thompson drugs
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
from ..imports.scRNA import ThompsonXA_SCGenes
from ..parafac2 import parafac2_nd
from ..decomposition import plotR2X
from ..crossVal import plotCrossVal


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((18, 25), (2, 4))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Drug, Cell, Gene]
    data = ThompsonXA_SCGenes(saveXA=False)

    # Performing parafac2 on single-cell Xarray
    _, factors, projs, _ = parafac2_nd(
        data["data"].to_numpy(),
        rank=13,
        verbose=True,
    )

    projs = xa.DataArray(
        projs,
        dims=["Drug", "Cell", "Cmp"],
        coords=dict(
            Drug=data.coords["Drug"],
            Cell=data.coords["Cell"],
            Cmp=[f"Cmp. {i}" for i in np.arange(1, projs.shape[2] + 1)],
        ),
        name="projections",
    )
    projs = xa.merge([projs, data["Cell Type"]], compat="no_conflicts")

    flattened_projs = projs.stack(AllCells=("Drug", "Cell"))

    # Remove empty slots
    nonzero_index = np.any(flattened_projs["projections"].to_numpy() != 0, axis=0)
    flattened_projs = flattened_projs.isel(AllCells=nonzero_index) 

    idxx = np.random.choice(
        len(flattened_projs.coords["AllCells"]), size=200, replace=False
    )
    
    plotFactors(factors, data["data"], ax[0:2], reorder=(0, 2), trim=(2,))
    
    plotSS(flattened_projs, ax[2])
    
    plotProj(flattened_projs.isel(AllCells=idxx), ax[3:5])

    plotR2X(data["data"].to_numpy(), 13, ax[5])
    
    plotCrossVal(data["data"].to_numpy(), 13, ax[6], trainPerc=0.75)

    return f
