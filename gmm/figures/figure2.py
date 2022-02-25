"""
This creates Figure 2.
"""
import numpy as np
import pandas as pd
import seaborn as sns


from .common import subplotLabel, getSetup
from ..imports import smallDF
from ..GMM import probGMM


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 4), (2, 3))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells wanted per experiment); 336 conditions in total
    cellperexp = 6000
    zflowDF, _ = smallDF(cellperexp)

    maxcluster = 5
    _, means, _ = probGMM(zflowDF, maxcluster, cellperexp)

    statMeans = means[:, :, 3]  # Expt x cluster

    meansDF = zflowDF.iloc[::cellperexp, :]  # Subset to one row per expt
    meansDF = meansDF[["Time", "Ligand", "Valency", "Dose"]]  # Only keep descriptive rows
    meansDF = pd.concat([meansDF] * maxcluster, ignore_index=True) # Duplicate for each cluster
    meansDF["Cluster"] = np.repeat(np.arange(1, maxcluster + 1), repeats=statMeans.shape[0]) # Track clusters
    meansDF["pSTAT5"] = statMeans.flatten(order="F")  # Checked the ordering here

    sns.scatterplot(data=meansDF, x="Dose", y="pSTAT5", hue="Cluster", ax=ax[4], style="Ligand")
    ax[4].set(xscale="log")

    return f
