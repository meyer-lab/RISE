"""
This creates Figure 1.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA


from .common import subplotLabel, getSetup
from ..imports import importflowDF, smallDF
from ..GMM import GMMpca

def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((8, 4), (2, 3))

    ax[5].axis("off")

    # Add subplot labels
    subplotLabel(ax)

    #smallDF(Amount of cells wanted per experiment)
    zflowDF = smallDF(100)

    arr = np.arange(1,5,1)
    totalvar = np.zeros([len(arr)])
    celltypelist = zflowDF.CellType.values
    totalDF = zflowDF.drop(columns=['CellType','pSTAT5'])

    # Determining variance explained 
    for a in range(len(arr)):
        pca = PCA(n_components=arr[a])
        pca.fit_transform(totalDF)
        totalvar[a] = sum(pca.explained_variance_ratio_)

    ax[0].scatter(arr,totalvar)
    xlabel = "Principal Components"
    ylabel = "Variance"
    ax[0].set(xlabel=xlabel, ylabel=ylabel)

    #This is genereal schematic of function GMMpca(ax,GMMcomparison,typeofGMM,zflowDF,maxcluster,ksplit)


    GMMpca(ax[1],"RandScore","pomegranate",zflowDF,6,5)
    GMMpca(ax[2],"Score","pomegranate",zflowDF,6,5)

    # GMMpca(ax[1],"RandScore","GMM",zflowDF,21,20)
    # GMMpca(ax[2],"Score","GMM",zflowDF,21,20)

    return f

