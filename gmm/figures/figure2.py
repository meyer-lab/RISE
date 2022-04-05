"""
This creates Figure 2.
"""
import seaborn as sns

from .common import subplotLabel, getSetup
from ..imports import smallDF
from ..GMM import probGMM


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 10), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells wanted per experiment): [DF] with all conditions as data
    cellperexp = 600
    zflowDF, _ = smallDF(cellperexp)

    # probGM(DF,maximum cluster,cellsperexperiemtn): [nk, means, covar] while using estimation gaussian parameters
    maxcluster = 4
    _, tMeans, _ = probGMM(zflowDF, maxcluster)

    # Tensorify data
    tMeansDF = tMeans.loc[:, "pSTAT5", :, :].to_dataframe("pSTAT5")
    tMeansDF = tMeansDF.reset_index()
    sns.scatterplot(data=tMeansDF, x="Dose", y="pSTAT5", hue="Cluster", ax=ax[0], style="Ligand")
    ax[0].set(xscale="log")

    heatmapDF = tMeansDF.copy()
    heatmapDF["Ligand/Dose"] = heatmapDF["Ligand"] + " - " + heatmapDF["Dose"].astype(str) + " (nM)"
    heatmapDF["Clust/Time"] = heatmapDF["Cluster"].astype(str) + " - " + heatmapDF["Time"].astype(str)
    heatmapDF = heatmapDF[["Ligand/Dose", "Clust/Time", "pSTAT5"]]
    heatmapDF = heatmapDF.pivot(index='Ligand/Dose', columns='Clust/Time', values='pSTAT5')
    sns.heatmap(heatmapDF, ax=ax[2])

    ax[3].hist(zflowDF["pSTAT5"].values, bins=1000, color='r')
    xlabel = "Event"
    ylabel = "pSTAT Signal"
    ax[3].set(xlabel=xlabel, ylabel=ylabel)

    wtntermDF = tMeans.loc[:, :, "WT C-term-1", :, :]

    for i, mark in enumerate(["Foxp3", "CD25", "CD45RA", "CD4", "pSTAT5"]):
        df = wtntermDF.loc[:, mark, :, :].to_dataframe(mark)
        sns.lineplot(data=df, x="Dose", y=mark, hue="Cluster", ax=ax[i + 4], palette='pastel', ci=None)
        ax[i + 4].set(xscale="log")

    return f
