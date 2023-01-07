"""
Parafac2 implementation on PBMCs treated across IL2 treatments, times, and doses
"""
import numpy as np
from .common import subplotLabel, getSetup, plotSCCP_factors
from ..imports.cytok import IL2_flowXA
from tensorly.decomposition import parafac2
from ..parafac2 import parafac2 as pf2

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # Import of single cells: [Ligand, Dose, Time, Cell, Marker]
    flowXA, _ = IL2_flowXA()
    
   # Performing parafac2 on single-cell Xarray
    rank = 5
    _, factors, _ = pf2(flowXA.to_numpy(), rank, n_iter_max=2000, 
                        tol=1e-8, nn_modes=None, verbose=False,
                        n_iter_parafac=5)
    
    plotSCCP_factors(rank, factors, flowXA, ax)

    return f
