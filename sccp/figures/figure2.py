"""
Parafac2 implementation on PBMCs treated across IL2 treatments, times, and doses
"""
import numpy as np
from .common import subplotLabel, getSetup, plotSCCP_factors
from ..imports.cytok import IL2_flowXA
from tensorly.decomposition import parafac2

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Gene, Cell, Drug]
    FlowXA, _ = IL2_flowXA(numCells=500)
    
    flattenFlowXA = np.reshape(FlowXA.to_numpy(), (FlowXA.shape[0], FlowXA.shape[1], -1))
    
    # Performing parafac2 on single-cell Xarray
    rank = 5
    weights, factors, _ = parafac2(
        flattenFlowXA,
        rank=rank,
        tol=1e-8,
        nn_modes=(0, 2),
        normalize_factors=True,
        verbose=True
    )

    plotSCCP_factors(rank, factors, FlowXA, ax)

    return f
