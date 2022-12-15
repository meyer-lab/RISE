"""
Investigating NK, covariance, and factors from tGMM for IL-2 dataset
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup, add_ellipse, plotCellAbundance
from gmm.imports import smallDF
from gmm.tensor import optimal_seed, cell_assignment, markerslist
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from .commonIL2 import (plotFactors_IL2,recapIL2, cluster_type)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 20), (8, 3))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells per experiment): Xarray of each marker, cell and condition
    # Final Xarray has dimensions [Marker, Cell Number, Time, Dose, Ligand]
    cellperexp = 300
    # marks = ["Foxp3", "CD25", "CD45RA", "CD4", "pSTAT5"]
    marks = ["Foxp3", "CD25", "pSTAT5"]
    flowXA, celltypeXA = smallDF(cellperexp)
    flowXA = flowXA.loc[marks, :, :, :, :]
    
    rank = 4; n_cluster = 7

    _, _, fit = optimal_seed(1, flowXA, rank=rank, n_cluster=n_cluster)
    fac = fit[0]

    plotCellAbundance(fac, n_cluster, ax[0])

    plotFactors_IL2(fac, flowXA, n_cluster, ax)
    
    time = 4.0; ligand = "WT C-term-1"
    recapIL2(fac, flowXA, time, ligand, marks, marks[0], marks[2], n_cluster, ax)

    cluster_type(fac, flowXA, celltypeXA[1], "Soft", ax[23])
    
    return f


