import numpy as np
import pandas as pd
import seaborn as sns
from .common import add_ellipse
from scCP.tensor import markerslist, cell_assignment
from sklearn.metrics import adjusted_rand_score, confusion_matrix

DimCol = [f"Dimension{i}" for i in np.arange(1, 6)]
    """Plots mean and covariance factors for ligand, time, dose, and cluster"""
    # Mean Factors 
    facXA = fac.get_factors_xarray(dataXA)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    markers = dataXA.coords[dataXA.dims[0]].values # Markers analyzed
    # Numb. of components
    xticks = facXA[DimCol[0]].coords[facXA[DimCol[0]].dims[1]].values
    # Lists for each dimension
    yticks = [[f"Clst: {i}" for i in np.arange(1, n_cluster + 1)],
                markers,
               [f"Time: {i}" for i in [1.0, 2.0, 4.0]],
               facXA[DimCol[3]].coords[facXA[DimCol[3]].dims[0]].values,
               facXA[DimCol[4]].coords[facXA[DimCol[4]].dims[0]].values]
    
    # Plotting mean factors
    for i, key in enumerate(facXA):
        data = facXA[key]
        sns.heatmap(
            data=data,
            xticklabels=xticks,
            yticklabels=yticks[i],
            ax=ax[i + 1], cmap=cmap,
            vmin=-1,vmax=1)
        ax[i + 1].set_xticklabels(ax[i+1].get_xticklabels(), rotation=90, ha="right")
        ax[i + 1].set_title("Mean Factors")
        ax[i + 1].tick_params(axis="y", rotation=0)

    # Covariance Factors
    cmap = sns.cubehelix_palette(as_cmap=True)
    cov_fac = fac.get_covariance_factors(dataXA) # Non-signal covariance factors
    covSig = cov_fac[DimCol[1]].to_numpy() # Signal covariance factors

    covfactors_place = [0, 2, 3, 4]
    
    # Plotting for covariance non-signal factors
    for i in range(len(covfactors_place)):
        sns.heatmap(data=cov_fac[DimCol[covfactors_place[i]]],
            xticklabels=xticks, yticklabels=yticks[covfactors_place[i]],
            ax=ax[i + 6], cmap=cmap, vmin=0, vmax=1)
        ax[i + 6].set_title("Covariance Factors")
        ax[i + 6].tick_params(axis="y", rotation=0)   
        
    # Plotting for covaraince signal factors
    for i in range(fac.rank):
        cov_signal = pd.DataFrame(
        covSig[:, :, i] @ covSig[:, :, i].T,
        columns=markers,
        index=markers)
        sns.heatmap(data=cov_signal, ax=ax[i + 10],cmap=cmap,
                    vmin=0, vmax=1)
        ax[i + 10].set(title="Covariance Factors: Rank - " + str(i + 1))
        ax[i + 10].tick_params(axis="y", rotation=0)   

def recapIL2(fac, flowXA, time, ligand, totalmarks, mark1, mark2, n_cluster, ax):
    """Comparing synthetic based data from output of tGMM to original IL-2 dataset"""
    timei = np.where(flowXA["Time"].values == time)[0][0]
    ligandi = np.where(flowXA["Ligand"].values == ligand)[0]
    marks = [mark1, mark2]
    
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
    
    for i in range(1, 4):
        points = np.squeeze(points_all[:, :, timei, i*3, ligandi]).T
        pointsDF = pd.DataFrame(
            {
                "Cluster": np.squeeze(points_y[:, timei, i*3, ligandi].astype(int)),
                mark1: points[:, totalmarks.index(mark1)],
                mark2: points[:, totalmarks.index(mark2)],
            }
        )
        sns.scatterplot(
            data=pointsDF,
            x=mark1,
            y=mark2,
            hue="Cluster",
            palette="tab10",
            ax=ax[15 + (i * 2)],
            s=3,alpha=.5
        )
        add_ellipse(
            timei,
            i*3,
            ligandi,
            fac,
            mark1,
            mark2,
            n_cluster,
            ax[15 + (i * 2)],
            colorpal,"IL2", totalmarks)
        ax[15 + (i * 2)].set(
            xlim=(-5, 5),
            ylim=(-5, 5),
            title=ligand
            + "-Time:"
            + str(time)
            + "-nM:"
            + str(flowXA.Dose.values[i*3])
            + "-ULTRA"
        )
        sns.scatterplot(
            data=markertotal.loc[markertotal["Dose"] == dose_unique[i*3]],
            x=mark1,
            y=mark2,
            ax=ax[(i * 2) + 16],
            s=3,alpha=.5
        )
        ax[(i * 2) + 16].set(
            xlim=(-5, 5),
            ylim=(-5, 5),
            title=ligand
            + "-Time:"
            + str(time)
            + "-nM:"
            + str(flowXA["Dose"].values[i*3])
            + "-Original Data",
        )
        
def cluster_type(fac, flowXA, typeXA, clusteringtype, ax):
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
        cmap = sns.light_palette("seagreen", as_cmap=True)
        sns.heatmap(data=clustDF, ax=ax,cmap=cmap)
        ax.set(title="Soft Clustering: Cell %")
