"""
Creating synthetic data and implementation of parafac2
"""
import numpy as np
from .common import (
    subplotLabel,
    getSetup,
    plotFactorsSynthetic,
    plotProj,
)
from ..synthetic import synthXA
from ..parafac2 import parafac2_nd
from ..decomposition import plotR2X
from ..crossVal import plotCrossVal


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    X = synthXA(magnitude=200, type="beach")

    # Performing parafac2 on single-cell Xarray
    _, factors, projs, _ = parafac2_nd(
        X,
        rank=2,
        verbose=True,
    )
    flattened_projs = np.concatenate(projs, axis=0)

    plotFactorsSynthetic(factors, X, ax[0:2])

    plotProj(projs[7], ax[2:4])

    plotProj(flattened_projs, ax[4:6])

    plotR2X(X, 3, ax[7])
    plotCrossVal(X.X_list, 3, ax[8], trainPerc=0.75)

    ax[2].set_title("Projections: Time=6")
    ax[4].set_title("Projections: All Conditions")
    ax[6].set_title("All Conditions")

    return f
