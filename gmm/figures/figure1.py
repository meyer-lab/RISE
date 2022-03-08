"""
This creates Figure 1.
"""
import pandas as pd
import seaborn as sns


from .common import subplotLabel, getSetup
from ..imports import smallDF
from ..GMM import cvGMM


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 4), (1, 3))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells wanted per experiment); 336 conditions in total
    cellperexp = 50
    zflowDF, experimentalcells = smallDF(cellperexp)

    ax[0].hist(experimentalcells, bins=20)
    xlabel = "Number of Cells per Experiment"
    ylabel = "Events"
    ax[0].set(xlabel=xlabel, ylabel=ylabel)

    # scoreDF(Determining rand_score and score for GMM with on DF and max cluster # with output [DF(Cluster #,Score)])
    maxcluster = 18
    scoreDF = cvGMM(zflowDF, maxcluster)

    for i in range(maxcluster):
        ax[1].plot(scoreDF.Cluster.values, scoreDF.rand_score.values)
        ax[2].plot(scoreDF.Cluster.values, scoreDF.ll_score.values)

    xlabel = "Cluster Number"
    ylabel = "Score"
    ax[1].set(xlabel=xlabel, ylabel=ylabel)
    ax[2].set(xlabel=xlabel, ylabel=ylabel)

    return f
