"""
Comparing synthetic based data from output of tGMM to original IL-2 dataset
"""
import enum
from optparse import Values
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from gmm.imports import smallDF
from gmm.tensor import minimize_func, gen_points_GMM, markerslist


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 8), (6, 4))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells per experiment): Xarray of each marker, cell and condition
    # Final Xarray has dimensions [Marker, Cell Number, Time, Dose, Ligand]
    cellperexp = 200
    zflowTensor, _ = smallDF(cellperexp, hyperlog=True)
    rank = 4
    n_cluster = 6
    print(np.amin(zflowTensor.data))

    time = 1.0
    ligand = "WT N-term-2"

    timei = np.where(zflowTensor.Time.values == time)[0][0]
    ligandi = np.where(zflowTensor.Ligand.values == ligand)[0]

    maximizedNK, _, optPTfactors, _, _, preNormOptCP = minimize_func(
        zflowTensor, rank=rank, n_cluster=n_cluster
    )

    markertotal = pd.DataFrame()
    for i, mark in enumerate(markerslist):
        markDF = zflowTensor.loc[mark, :, time, :, ligand]
        markDF = markDF.to_dataframe(mark).reset_index()
        markertotal[mark] = markDF[mark].values

    markertotal["Dose"] = markDF["Dose"].values
    markertotal["Cell"] = markDF["Cell"].values

    dose_unique = np.unique(markDF["Dose"].values)

    for dose in range(0, 12):
        points = gen_points_GMM(
            maximizedNK,
            preNormOptCP,
            optPTfactors,
            timei,
            dose,
            ligandi,
            n_samples=cellperexp,
        )
        pointsDF = pd.DataFrame(
            {
                "Cluster": points[1],
                "Foxp3": points[0][:, 0],
                "CD25": points[0][:, 1],
                "CD45RA": points[0][:, 2],
                "CD4": points[0][:, 3],
                "pSTAT5": points[0][:, 4],
            }
        )
        sns.scatterplot(
            data=pointsDF,
            x="pSTAT5",
            y="CD25",
            hue="Cluster",
            palette="tab10",
            ax=ax[dose * 2],
            s=5,
        )
        sns.scatterplot(
            data=markertotal.loc[markertotal["Dose"] == dose_unique[dose]],
            x="pSTAT5",
            y="CD25",
            ax=ax[(dose * 2) + 1],
            s=5,
        )
        ax[dose * 2].set(
            xlim=(-5, 5),
            ylim=(-5, 5),
            title=ligand
            + " at time "
            + str(time)
            + " at nM="
            + str(zflowTensor.Dose.values[dose]),
        )
        ax[(dose * 2) + 1].set(
            xlim=(-5, 5),
            ylim=(-5, 5),
            title=ligand
            + " at time "
            + str(time)
            + " at nM="
            + str(zflowTensor.Dose.values[dose])
            + ":Original Data",
        )

    return f
