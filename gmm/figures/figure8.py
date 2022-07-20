"""
Investigating synthetic data based on results from tGMM for IL-2 dataset
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from gmm.imports import smallDF
from gmm.tensor import minimize_func, gen_points_GMM


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 8), (3, 4))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells per experiment): Xarray of each marker, cell and condition
    # Final Xarray has dimensions [Marker, Cell Number, Time, Dose, Ligand]
    cellperexp = 200
    zflowTensor, _ = smallDF(cellperexp)
    rank = 4
    n_cluster = 6
    time = 1.0
    ligand = "IL2-1"

    timei = np.where(zflowTensor.Time.values == time)[0][0]
    ligandi = np.where(zflowTensor.Ligand.values == ligand)[0]

    maximizedNK, _, optPTfactors, _, _, preNormOptCP = minimize_func(zflowTensor, rank=rank, n_cluster=n_cluster)

    for dose in range(0, 12):
        points = gen_points_GMM(maximizedNK, preNormOptCP, optPTfactors, timei, dose, ligandi)
        pointsDF = pd.DataFrame({"Cluster": points[1],'Foxp3': points[0][:, 0], 'CD25': points[0][:, 1], 'CD45RA': points[0][:, 2], 'CD4': points[0][:, 3], 'pSTAT5': points[0][:, 4]})
        sns.scatterplot(data=pointsDF, x="Foxp3", y="pSTAT5", hue="Cluster", palette="tab10", ax=ax[dose])
        ax[dose].set(xlim=(-5, 5), ylim=(-5, 5), title=ligand + " at time " + str(time) + " at nM=" + str(zflowTensor.Dose.values[dose]))

    return f
