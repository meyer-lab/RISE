"""
This creates Figure 4.
"""
import numpy as np
import tensorly as tl
from scipy.optimize import minimize
from jax.config import config
from jax import value_and_grad
from .common import subplotLabel, getSetup
from ..imports import smallDF
from ..GMM import probGMM
from ..tensor import tensor_decomp, cp_to_vector, tensorcovar_decomp, pt_to_vector, maxloglik_ptnnp


config.update("jax_enable_x64", True)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells per experiment): Xarray of each marker, cell and condition
    # Final Xarray has dimensions [Marker, Cell Number, Time, Dose, Ligand]
    cellperexp = 50
    zflowTensor, _ = smallDF(cellperexp)

    # probGM(Xarray, max cluster): Xarray [nk, means, covar] while using estimation gaussian parameters
    # tMeans[Cluster, Marker, Time, Dose, Ligand]
    maxcluster = 6
    nk, tMeans, tPrecision = probGMM(zflowTensor, maxcluster)

    # tensorcovar_decomp(precision, rank, nk):
    # [DF,core tensor, tensorfactors] creates DF of factors for different
    # conditions and output of decomposition

    ranknumb = 3
    _, facInfo = tensor_decomp(tMeans, ranknumb, "NNparafac")

    ptFactors, ptCore = tensorcovar_decomp(tPrecision, ranknumb)

    vecptFacCore, ptFacLength = pt_to_vector(ptFactors, ptCore)

    nkValues = np.exp(np.nanmean(np.log(nk), axis=(1, 2, 3)))
    cpVector = cp_to_vector(facInfo)
    totalVector = np.concatenate((nkValues, cpVector, vecptFacCore))

    args = (tPrecision, facInfo, zflowTensor, len(cpVector), ptFacLength, ptCore)

    tl.set_backend("jax")

    func = value_and_grad(maxloglik_ptnnp)

    opt = minimize(func, totalVector, jac=True, method="L-BFGS-B", args=args, options={"iprint": 50, "maxiter": 1000})

    tl.set_backend("numpy")

    return f
