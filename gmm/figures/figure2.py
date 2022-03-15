"""
This creates Figure 2.
"""
import numpy as np
import pandas as pd
import seaborn as sns
# import tensorly as tl


from .common import subplotLabel, getSetup
from ..imports import smallDF
from ..GMM import probGMM


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 30), (3, 3))

    # Add subplot labels
    subplotLabel(ax)

    #smallDF(Amount of cells wanted per experiment)
    cellperexp = 6000
    zflowDF, _ = smallDF(cellperexp)

    maxcluster = 5
    nk, means, covar = probGMM(zflowDF, maxcluster, cellperexp)

    meansDF = zflowDF.iloc[::cellperexp, :]  # Subset to one row per expt
    meansDF = meansDF[["Time", "Ligand", "Valency", "Dose"]]  # Only keep descriptive rows
    meansDF = pd.concat([meansDF] * maxcluster, ignore_index=True)  # Duplicate for each cluster
    markerslist = ["Foxp3", "CD25", "CD45RA", "CD4","pSTAT5"]
    for i,mark in enumerate(markerslist):
        markers_means = means[:,:,i]
        meansDF[mark] = markers_means.flatten(order="F") 

    meansDF["Cluster"] = np.repeat(np.arange(1, maxcluster + 1), repeats= markers_means.shape[0])  # Track clusters
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

            heatmapDF = pd.concat([heatmapDF,row])
    
    heatmapDF = heatmapDF.set_index("Ligand/Dose")
    sns.heatmap(heatmapDF, ax=ax[2])

    ax[3].hist(zflowDF["pSTAT5"].values,bins=10000)
    ax[3].set(xlim=(0, 40000))

    xlabel = "Event"
    ylabel = "pSTAT Signal"
    ax[3].set(xlabel=xlabel, ylabel=ylabel)

    wtntermDF  = meansDF.loc[meansDF["Ligand"] == "WT C-term"]

    for i, mark in enumerate(markerslist):
        sns.lineplot(data=wtntermDF, x="Dose", y=mark, hue="Cluster", ax=ax[i+4],palette='pastel',ci= None)
        ax[i+4].set(xscale="log")

    meansDF = meansDF.drop("NK", axis=1)
    meansDF = meansDF.set_index(["Ligand", "Dose", "Time", "Cluster", "Valency"])
    xtensor = meansDF.to_xarray()
    tensor = xtensor.to_array(dim="Marker")
    print(tensor)

    return f
