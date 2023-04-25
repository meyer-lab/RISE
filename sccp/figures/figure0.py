"""
Creating synthetic data and implementation of parafac2
"""
import numpy as np
from .common import (
    subplotLabel,
    getSetup,
    plotFactorsSynthetic,
    plotProj,
    plotR2X,
    plotCV
)
from ..synthetic import synthXA
from ..parafac2 import parafac2_nd


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    X = synthXA(magnitude=200, type="beach")

    rank=2
    # Performing parafac2 on single-cell Xarray
    _, factors, projs, _ = parafac2_nd(
        X,
        rank=rank,
        random_state=1,
    )
    flattened_projs = np.concatenate(projs, axis=0)

    plotFactorsSynthetic(factors, X, ax[0:2])

    plotProj(projs[7], ax[2:4])

    plotProj(flattened_projs, ax[4:6])
    
    plotCV(X, rank+3, trainPerc=0.75, ax=ax[7])
    plotR2X(X, rank+3, ax=ax[8])
    
    ax[2].set_title("Projections: Time=6")
    ax[4].set_title("Projections: All Conditions")
    ax[6].set_title("All Conditions")

    return f
