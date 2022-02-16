"""
This creates Figure 1.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from pathlib import Path

from .common import subplotLabel, getSetup
from ..imports import importflowDF, smallDF
from ..GMM import GMMpca, runPCA


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    ax, f = getSetup((8, 4), (2, 3))

    ax[5].axis("off")

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells wanted per experiment)
    zflowDF = smallDF(50)

    # PCA(Runs PCA on dataframe with output [PCs,VarianceExplained])
    components, vcexplained = runPCA(zflowDF)

    ax[0].scatter(components, vcexplained)
    xlabel = "Principal Components"
    ylabel = "Variance"
    ax[0].set(xlabel=xlabel, ylabel=ylabel)

    # Determining rand_score for GMM with dataframe
    randDF = GMMpca(zflowDF, 10, "rand_score")
    scoreDF = GMMpca(zflowDF, 10)

    for i in range(len(components)):
        ax[1].plot(randDF.Cluster.values, randDF.Score.values)
        ax[2].plot(scoreDF.Cluster.values, scoreDF.Score.values)

    ax[1].legend(title="Component Number", loc='best')

    xlabel = "Cluster Number"
    ylabel = "Score"
    ax[1].set(xlabel=xlabel, ylabel=ylabel)
    ax[2].legend(title="Component Number", loc='best')
    ax[2].set(xlabel=xlabel, ylabel=ylabel)

    # filepath = Path('gmm/output/figure1.csv')
    # filepath.parent.mkdir(parents=True, exist_ok=True)
    # zflowDF.to_csv(filepath)

    return f
