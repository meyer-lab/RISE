"""
Comparing synthetic based data from output of tGMM to original IL-2 dataset
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup, add_ellipse
from gmm.imports import smallDF
from gmm.tensor import minimize_func, optimal_seed


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (6, 4))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells per experiment): Xarray of each marker, cell and condition
    # Final Xarray has dimensions [Marker, Cell Number, Time, Dose, Ligand]
    cellperexp = 300
    marks = ["Foxp3","CD25","pSTAT5"]
    flowXA, _ = smallDF(cellperexp)
    flowXA = flowXA.loc[marks,:,:,:,:]
    rank = 3
    n_cluster = 4
    assert np.amin(flowXA.data) < 0

    time = 4.0
    ligand = "WT C-term-1"

    timei = np.where(flowXA["Time"].values == time)[0][0]
    ligandi = np.where(flowXA["Ligand"].values == ligand)[0]

    _, _, fit = optimal_seed(5, flowXA, rank=rank, n_cluster=n_cluster)
    fac = fit[0]

    markertotal = pd.DataFrame()
    for mark in marks:
        markDF = flowXA.loc[mark, :, time, :, ligand]
        markDF = markDF.to_dataframe(mark).reset_index()
        markertotal[mark] = markDF[mark].values

    markertotal["Dose"] = markDF["Dose"].values
    markertotal["Cell"] = markDF["Cell"].values
    dose_unique = np.unique(markDF["Dose"].values)

    colorpal = sns.color_palette("tab10", n_cluster)

    points_all, points_y = fac.sample(n_samples=500)

    for dose in range(0, 12):
        points = np.squeeze(points_all[:, :, timei, dose, ligandi]).T
        pointsDF = pd.DataFrame(
            {
                "Cluster": np.squeeze(points_y[:, timei, dose, ligandi]),
                "Foxp3": points[:, 0],
                "CD25": points[:, 1],
                "pSTAT5": points[:, 2]
            }
        )
        sns.scatterplot(
            data=pointsDF,
            x=marks[1],
            y=marks[2],
            hue="Cluster",
            palette="tab10",
            ax=ax[dose * 2],
            s=3,alpha=.5
        )
        add_ellipse(
            timei,
            dose,
            ligandi,
            fac,
            marks[1],
            marks[2],
            n_cluster,
            ax[dose * 2],
            colorpal,"IL2",marks)
        sns.scatterplot(
            data=markertotal.loc[markertotal["Dose"] == dose_unique[dose]],
            x=marks[1],
            y=marks[2],
            ax=ax[(dose * 2) + 1],
            s=3,alpha=.5
        )
        ax[dose * 2].set(
            xlim=(-5, 5),
            ylim=(-5, 5),
            title=ligand
            + "-Time:"
            + str(time)
            + "-nM:"
            + str(flowXA.Dose.values[dose])
            + "-ULTRA"
        )
        ax[(dose * 2) + 1].set(
            xlim=(-5, 5),
            ylim=(-5, 5),
            title=ligand
            + "-Time:"
            + str(time)
            + "-nM:"
            + str(flowXA["Dose"].values[dose])
            + "-Original Data",
        )

    return f
