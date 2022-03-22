"""
This creates Figure 2.
"""
import numpy as np
import pandas as pd
import seaborn as sns

from .common import subplotLabel, getSetup
from ..imports import smallDF
from ..GMM import probGMM


def meanmarkerDF(zflowDF, cellperexp, means, nk, maxcluster):
    """Combines NK/Mean Values into DF and correspond to different conditions per clusters"""
    meansDF = zflowDF.iloc[::cellperexp, :]  # Subset to one row per expt
    meansDF = meansDF[["Time", "Ligand", "Valency", "Dose"]]  # Only keep descriptive rows
    meansDF = pd.concat([meansDF] * maxcluster, ignore_index=True)  # Duplicate for each cluster
    markerslist = ["Foxp3", "CD25", "CD45RA", "CD4", "pSTAT5"]
    for i, mark in enumerate(markerslist):
        markers_means = means[:, :, i]
        meansDF[mark] = markers_means.flatten(order="F")

    meansDF["Cluster"] = np.repeat(np.arange(1, maxcluster + 1), repeats=markers_means.shape[0])  # Track clusters
    meansDF["NK"] = nk.flatten(order="F")

    return meansDF, markerslist


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 4), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells wanted per experiment): [DF] with all conditions as data
    cellperexp = 600
    zflowDF, _ = smallDF(cellperexp)

    # probGM(DF,maximum cluster,cellsperexperiemtn): [nk, means, covar] while using estimation gaussian parameters
    maxcluster = 4
    nk, means, _ = probGMM(zflowDF, maxcluster, cellperexp)

    # meanmarkerDF(DF,cells per experiment, mean values, nk values, maximum cluster): [DF, diff. marker list] inputs means/NK into DF
    meansDF, markerslist = meanmarkerDF(zflowDF, cellperexp, means, nk, maxcluster)

    sns.scatterplot(data=meansDF, x="Dose", y="pSTAT5", hue="Cluster", ax=ax[0], style="Ligand")
    ax[0].set(xscale="log")

    sns.scatterplot(data=meansDF, x="Dose", y="NK", hue="Cluster", ax=ax[1], style="Ligand")
    ax[1].set(xscale="log")

    heatmapDF = meansDF.copy()
    heatmapDF["Ligand/Dose"] = heatmapDF["Ligand"] + " - " + heatmapDF["Dose"].astype(str) + " (nM)"
    heatmapDF["Clust/Time"] = heatmapDF["Cluster"].astype(str) + " - " + heatmapDF["Time"].astype(str)
    heatmapDF = heatmapDF[["Ligand/Dose", "Clust/Time", "pSTAT5"]]
    heatmapDF = heatmapDF.pivot(index='Ligand/Dose', columns='Clust/Time', values='pSTAT5')
    sns.heatmap(heatmapDF, ax=ax[2])

    ax[3].hist(zflowDF["pSTAT5"].values, bins=1000, color='r')
    xlabel = "Event"
    ylabel = "pSTAT Signal"
    ax[3].set(xlabel=xlabel, ylabel=ylabel)

    wtntermDF = meansDF.loc[meansDF["Ligand"] == "WT C-term"]

    for i, mark in enumerate(markerslist):
        sns.lineplot(data=wtntermDF, x="Dose", y=mark, hue="Cluster", ax=ax[i + 4], palette='pastel', ci=None)
        ax[i + 4].set(xscale="log")

    return f
