"""
This creates Figure 4.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import tensorly as tl
from scipy.optimize import minimize
from jax.config import config
from jax import value_and_grad
from .common import subplotLabel, getSetup
from gmm.imports import smallDF
from gmm.GMM import probGMM
from gmm.tensor import tensor_decomp, tensorcovar_decomp, cp_pt_to_vector, maxloglik_ptnnp, vector_to_cp_pt
from tensorly.cp_tensor import cp_normalize


config.update("jax_enable_x64", True)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (2, 3))

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

    _, ptCore = tensorcovar_decomp(tPrecision, ranknumb)

    facVector = cp_pt_to_vector(facInfo, ptCore)
    nkValues = np.exp(np.nanmean(np.log(nk), axis=(1, 2, 3)))
    totalVector = np.concatenate((nkValues, facVector))

    args = (facInfo, zflowTensor)

    tl.set_backend("jax")

    func = value_and_grad(maxloglik_ptnnp)

    opt = minimize(func, totalVector, jac=True, method="L-BFGS-B", args=args, options={"iprint": 50, "maxiter": 1000})

    tl.set_backend("numpy")

    rebuildCpFactors, rebuildPtFactors, rebuildPtCore = vector_to_cp_pt(opt.x[facInfo.shape[0]::], facInfo.rank, facInfo.shape)
    maximizedCpInfo = cp_normalize(rebuildCpFactors)

    ax[0].bar(np.arange(1,maxcluster+1),opt.x[0:facInfo.shape[0]])
    xlabel = "Cluster"
    ylabel = "NK Value"
    ax[0].set(xlabel=xlabel, ylabel=ylabel)


    cmpCol = [f"Cmp. {i}" for i in np.arange(1, ranknumb + 1)]

    maximizedFactors = []
    for ii, dd in enumerate(tMeans.dims):
        maximizedFactors.append(pd.DataFrame(maximizedCpInfo.factors[ii], columns=cmpCol, index=tMeans.coords[dd]))

    for i in range(0, len(facInfo.shape)):
        heatmap = sns.heatmap(data= maximizedFactors[i], ax=ax[i+1], vmin=0, vmax=1, cmap="Blues")


    return f
