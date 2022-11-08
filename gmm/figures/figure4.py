"""
Investigating NK, covariance, and factors from tGMM for IL-2 dataset
"""
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from gmm.imports import smallDF
from gmm.tensor import optimal_seed, cell_assignment
from sklearn.metrics import adjusted_rand_score, confusion_matrix


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 20), (7, 3))

    # Add subplot labels
    subplotLabel(ax)

    # smallDF(Amount of cells per experiment): Xarray of each marker, cell and condition
    # Final Xarray has dimensions [Marker, Cell Number, Time, Dose, Ligand]
    cellperexp = 300
    marks = ["Foxp3", "CD25", "pSTAT5"]
    flowXA, celltypeXA = smallDF(cellperexp)
    flowXA = flowXA.loc[marks, :, :, :, :]

    rank = 6
    n_cluster = 8

    _, _, fit = optimal_seed(5, flowXA, rank=rank, n_cluster=n_cluster)
    fac = fit[0]

    ax[0].bar(np.arange(1, fac.nk.size + 1), fac.norm_NK(), color="k")
    ax[0].set(xlabel="Cluster", ylabel="Cell Abundance")

    # CP factors
    facXA = fac.get_factors_xarray(flowXA)

    plotFactors_IL2(facXA, ax)
    plotCovarFactors_IL2(fac, flowXA, n_cluster, ax)

    cluster_type(flowXA, fac, celltypeXA[1], ax[20], "Soft")

    return f


def plotFactors_IL2(facXA, ax):
    """Plots factors for ligand, time, dose, and cluster"""
    for i, key in enumerate(facXA):
        data = facXA[key]
        sns.heatmap(
            data=data,
            xticklabels=data.coords[data.dims[1]].values,
            yticklabels=data.coords[data.dims[0]].values,
            ax=ax[i + 1],
        )
        ax[i + 1].set_title("Mean Factors")


def plotCovarFactors_IL2(fac, dataXA, n_cluster, ax):
    """Plots covariance factors for ligand, time, dose, and cluster"""
    cov_fac = fac.get_covariance_factors(dataXA)
    DimCol = [f"Dimension{i}" for i in np.arange(1, len(cov_fac) + 1)]
    covSig = cov_fac[DimCol[1]].to_numpy()

    for i in range(fac.rank):
        cov_signal = pd.DataFrame(
        covSig[:, :, i] @ covSig[:, :, i].T,
        columns=dataXA.coords[dataXA.dims[0]],
        index=dataXA.coords[dataXA.dims[0]])
        sns.heatmap(data=cov_signal, ax=ax[i + 10])
        ax[i + 10].set(title="Covariance: Rank - " + str(i + 1))

    for i in range(len(fac.covFacs)):
        if i == 0:
            sns.heatmap(
                data=fac.covFacs[i],
                xticklabels=cov_fac[DimCol[i]].coords[cov_fac[DimCol[i]].dims[1]].values,
                yticklabels=cov_fac[DimCol[i]].coords[cov_fac[DimCol[i]].dims[0]].values,
                ax=ax[i + 6],
            )
        else:
            sns.heatmap(
                data=fac.covFacs[i],
                xticklabels=cov_fac[DimCol[i+1]].coords[cov_fac[DimCol[i+1]].dims[1]].values,
                yticklabels=cov_fac[DimCol[i+1]]
                .coords[cov_fac[DimCol[i+1]].dims[0]]
                .values,
                ax=ax[i + 6],
            )
        ax[i + 6].set_title("Covariance Factors")

    return


def cluster_type(flowXA, fac, typeXA, ax, clusteringtype):
    """Solves for confusion matrix of predicted and actual cell type labels"""
    # Solves for cell assignments shape [Cell, Cell Prob (Clust), Time, Dose, Ligand]
    resps = cell_assignment(flowXA.to_numpy(), fac)
    # Normalizes each responsibility for every cell to equal 1 for every condition
    resps = resps / np.reshape(
        np.sum(resps, axis=1),
        (-1, 1, flowXA.shape[2], flowXA.shape[3], flowXA.shape[4]),
    )

    # Finding hard clustering assignment for each cell
    tensor_pred = np.argmax(resps, axis=1) + 1

    # Comparing between predicted and actual assignments of all cells
    print(
        "Rand_Score:",
        adjusted_rand_score(
            np.ravel(typeXA.to_numpy()), np.ravel(tensor_pred.astype(int))
        ),
    )
    # 1.0 is a perfect match for rand_score

    if clusteringtype == "Hard":
        confmatrix = confusion_matrix(
            np.ravel(typeXA.to_numpy()), np.ravel(tensor_pred.astype(int))
        )
        confmatrix = confmatrix[
            0 : len(np.unique(typeXA.to_numpy())), 0 : resps.shape[1]
        ]
        confDF = pd.DataFrame(
            data=confmatrix,
            index=["None", "Treg", "Thelper"],
            columns=[f"Clst. {i}" for i in np.arange(1, resps.shape[1] + 1)],
        )
        sns.heatmap(data=confDF, ax=ax, annot=True)
        ax.set(title="Hard Clustering: Confusion Matrix")

    else:
        type_tensor = typeXA.to_numpy()
        clustDF = pd.DataFrame()
        celltype_dict = ["None", "Treg", "Thelper"]
        # Iteratives over each cell type and cluster, sums the normalized responsibilites of partial clustering
        for i in range(len(celltype_dict)):
            truecell = type_tensor == i + 1
            truecell_index = np.argwhere(truecell == True)
            totalresps = np.sum(
                resps[
                    truecell_index[:, 0],
                    :,
                    truecell_index[:, 1],
                    truecell_index[:, 2],
                    truecell_index[:, 3],
                ],
                axis=0,
            )
            for j in range(resps.shape[1]):
                clustDF = pd.concat(
                    [
                        clustDF,
                        pd.DataFrame(
                            {
                                "Cluster": [j + 1],
                                "Cell Type": [celltype_dict[i]],
                                "Total Resp": np.asarray(totalresps[j]),
                            }
                        ),
                    ]
                )

        clustDF = clustDF.reset_index(drop=True)
        clustDF = clustDF.pivot(
            index="Cell Type", columns="Cluster", values="Total Resp"
        )
        clustDF = clustDF.div(clustDF.sum(axis=0))
        assert np.isfinite(clustDF.to_numpy().all())
        sns.heatmap(data=clustDF, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set(title="Soft Clustering: Cell %")
