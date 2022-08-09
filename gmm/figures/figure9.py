"""
Comparing synthetic based data from output of tGMM to original IL-2 dataset
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup, add_ellipse
from gmm.imports import smallDF
from gmm.tensor import minimize_func, markerslist


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 10), (6, 4))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells per experiment): Xarray of each marker, cell and condition
    # Final Xarray has dimensions [Marker, Cell Number, Time, Dose, Ligand]
    cellperexp = 100
    zflowTensor, _ = smallDF(cellperexp)
    rank = 3
    n_cluster = 3
    assert np.amin(zflowTensor.data) < 0

    time = 1.0
    ligand = "WT N-term-2"

    timei = np.where(zflowTensor.Time.values == time)[0][0]
    ligandi = np.where(zflowTensor.Ligand.values == ligand)[0]

    fac, _, _ = minimize_func(
        zflowTensor, rank=rank, n_cluster=n_cluster
    )

    markertotal = pd.DataFrame()
    for mark in markerslist:
        markDF = zflowTensor.loc[mark, :, time, :, ligand]
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
                "CD45RA": points[:, 2],
                "CD4": points[:, 3],
                "pSTAT5": points[:, 4],
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
        add_ellipse(
            timei,
            dose,
            ligandi,
            fac,
            "pSTAT5",
            "CD25",
            n_cluster,
            ax[dose * 2],
            colorpal,
            datatype="IL2",
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
            + " at Time "
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
