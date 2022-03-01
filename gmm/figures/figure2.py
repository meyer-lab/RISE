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
    ax, f = getSetup((20, 30), (2, 3))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells wanted per experiment); 336 conditions in total
    cellperexp = 6000
    zflowDF, _ = smallDF(cellperexp)

    maxcluster = 5
    nk, means, _ = probGMM(zflowDF, maxcluster, cellperexp)

    statMeans = means[:, :, 4]  # Expt x cluster x pSTAT5

    meansDF = zflowDF.iloc[::cellperexp, :]  # Subset to one row per expt
    meansDF = meansDF[["Time", "Ligand", "Valency", "Dose"]]  # Only keep descriptive rows
    meansDF = pd.concat([meansDF] * maxcluster, ignore_index=True)  # Duplicate for each cluster
    meansDF["Cluster"] = np.repeat(np.arange(1, maxcluster + 1), repeats=statMeans.shape[0])  # Track clusters
    meansDF["pSTAT5"] = statMeans.flatten(order="F")  # Checked the ordering here
    meansDF["NK"] = nk.flatten(order="F")

    sns.scatterplot(data=meansDF, x="Dose", y="pSTAT5", hue="Cluster", ax=ax[0], style="Ligand")
    ax[0].set(xscale="log")

    sns.scatterplot(data=meansDF, x="Dose", y="NK", hue="Cluster", ax=ax[1], style="Ligand")
    ax[1].set(xscale="log")

    heatmap = meansDF
    heatmapDF = pd.DataFrame()

    for ligand in heatmap.Ligand.unique():
        for dose in heatmap.Dose.unique():
            row = pd.DataFrame()
            row["Ligand/Dose"] = [ligand + " - " + str(dose) + " (nM)"]
            for tim in heatmap.Time.unique():
                for clust in heatmap.Cluster.unique():
                    entry = heatmap.loc[(heatmap.Ligand == ligand) & (heatmap.Dose == dose) & (heatmap.Cluster == clust) & (heatmap.Time == tim)]
                    row["Cluster:" + str(clust) + " - " + str(tim) + " hrs"] = entry.pSTAT5.to_numpy()

            heatmapDF = heatmapDF.append(row)

    heatmapDF = heatmapDF.set_index("Ligand/Dose")
    sns.heatmap(heatmapDF, ax=ax[2])

    return f
