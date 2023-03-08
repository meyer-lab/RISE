"""
This file contains functions that are used in multiple figures.
"""
from string import ascii_lowercase
import sys
import time
import seaborn as sns
import pandas as pd
import matplotlib
from matplotlib import gridspec, pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.metrics import silhouette_samples

matplotlib.use("AGG")

matplotlib.rcParams["legend.labelspacing"] = 0.2
matplotlib.rcParams["legend.fontsize"] = 8
matplotlib.rcParams["xtick.major.pad"] = 1.0
matplotlib.rcParams["ytick.major.pad"] = 1.0
matplotlib.rcParams["xtick.minor.pad"] = 0.9
matplotlib.rcParams["ytick.minor.pad"] = 0.9
matplotlib.rcParams["legend.handletextpad"] = 0.5
matplotlib.rcParams["legend.handlelength"] = 0.5
matplotlib.rcParams["legend.framealpha"] = 0.5
matplotlib.rcParams["legend.markerscale"] = 0.7
matplotlib.rcParams["legend.borderpad"] = 0.35
matplotlib.rcParams["svg.fonttype"] = "none"


def getSetup(figsize, gridd, multz=None, empts=None, constrained_layout=True):
    """Establish figure set-up with subplots."""
    sns.set(
        style="whitegrid",
        font_scale=0.7,
        color_codes=True,
        palette="colorblind",
        rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6},
    )

    # create empty list if empts isn't specified
    if empts is None:
        empts = []

    if multz is None:
        multz = {}

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, constrained_layout=constrained_layout)
    gs1 = gridspec.GridSpec(*gridd, figure=f)

    # Get list of axis objects
    x = 0
    ax = []
    while x < gridd[0] * gridd[1]:
        if x not in empts and x not in multz.keys():  # If this is just a normal subplot
            ax.append(f.add_subplot(gs1[x]))
        elif x in multz.keys():  # If this is a subplot that spans grid elements
            ax.append(f.add_subplot(gs1[x : x + multz[x] + 1]))
            x += multz[x]
        x += 1

    return (ax, f)


def subplotLabel(axs):
    """Place subplot labels on figure."""
    for ii, ax in enumerate(axs):
        ax.text(
            -0.2,
            1.2,
            ascii_lowercase[ii],
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )


def genFigure():
    """Main figure generation function."""
    fdir = "./output/"
    start = time.time()
    nameOut = "figure" + sys.argv[1]

    exec("from sccp.figures." + nameOut + " import makeFigure", globals())
    ff = makeFigure()
    ff.savefig(fdir + nameOut + ".svg", dpi=300, bbox_inches="tight", pad_inches=0)

    print(f"Figure {sys.argv[1]} is done after {time.time() - start} seconds.\n")


def plotFactorsSynthetic(factors, data_xarray, ax):
    """Plots parafac2 factors for synthetic data"""
    rank = factors[0].shape[1]
    xticks = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    iter = 0
    for i in range(0, len(factors)):
        if i != len(factors)-2:
            if i == 0:
                timeDF = pd.DataFrame(factors[i], columns = xticks)
                timeDF["Time"] = np.arange(1, data_xarray.shape[i] + 1)
                sns.lineplot(data=timeDF[xticks], ax=ax[iter])
                ax[iter].set(
                    ylabel="Cmp. Weight",
                    xlabel="Time",
                    xticks=np.arange(0, data_xarray.shape[i]))
                
            else:       
                yt = data_xarray.coords[data_xarray.dims[i]].values
                X = factors[i]
                sns.heatmap(
                    data=X,
                    xticklabels=xticks,
                    yticklabels=yt,
                    ax=ax[iter],
                    center=0,
                    cmap=cmap,
                )

            ax[iter].set_title("Factors")
            ax[iter].tick_params(axis="y", rotation=0)
            iter += 1

                    
def plotFactors(factors, data_xarray, ax, reorder=tuple(), trim=tuple()):
    """Plots parafac2 factors for synthetic data"""
    rank = factors[0].shape[1]
    xticks = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    iter = 0
    for i in range(0, len(factors)):
        # The single cell mode has a square factors matrix
        if i != len(factors)-2:
            yt = data_xarray.coords[data_xarray.dims[i]].values
            X = factors[i]

            if i in trim:
                max_weight = np.max(np.abs(X), axis=1)
                kept_idxs = max_weight > 0.08
                X = X[kept_idxs]
                yt = yt[kept_idxs]

            if i in reorder:
                X, ind = reorder_table(X)
                yt = yt[ind]

            sns.heatmap(
                data=X,
                xticklabels=xticks,
                yticklabels=yt,
                ax=ax[iter],
                center=0,
                cmap=cmap,
            )

            ax[iter].set_title("Factors")
            ax[iter].tick_params(axis="y", rotation=0)
            iter += 1
              
            if i == 2 and len(yt) > 50:
                sort_idx = np.argsort(X, axis=0)
                for j in range(rank):
                    sort_data = yt[sort_idx[:, j]]
                    print("Bottom 10 Genes Cmp." + str(j+1) + ":", sort_data[:10])
                    print("Top 10 Genes Cmp." + str(j+1) + ":", np.flip(sort_data[-10:]))  
    
def plotProjs_SS(factors, projs, celltypeXA, color_palette, ax, size=100):
    """Plots parafac2 projections matrix with compenent weights and silhoutte scores"""
    rank = factors[0].shape[1]
    xticks = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    silhouetteDF =  pd.DataFrame([])
    for i, ps in enumerate(projs):
        ps = np.dot(ps, factors[-2])
        nonzero = ~np.all(ps == 0, axis=1)
        pps = ps[nonzero]
        ctDF = celltypeXA[i,nonzero].to_dataframe().reset_index()
        ctDF.sort_values(by=["Cell Type"], inplace=True)
        ind = ctDF.index.values
        reordered_projs = pps[ind]
        random_index = np.sort(np.random.choice(reordered_projs.shape[0], size=size, replace=False))
    
        sns.heatmap(
            data=np.flip(reordered_projs[random_index],axis=0),
            xticklabels=xticks,
            yticklabels=False,
            center=0,
            ax=ax[2*i + len(factors)-1],
            cmap=cmap,
        )
    
        true_celltypes = ctDF.loc[ind].set_index("Cell Type")
        celltypesDF = true_celltypes.drop(columns=true_celltypes.columns)
        allcelltypes = celltypesDF.copy()
        celltypesDF["Cell Type"] = 0
        label_colorbar = []
        choose_color_palette = []
        colorbar_numbers = np.arange(0, len(np.unique(allcelltypes.index)))
        for j, label in enumerate(np.unique(allcelltypes.index)):
            celltypesDF[celltypesDF.index == label] = j
            choose_color_palette = np.append(choose_color_palette, color_palette[j])
            label_colorbar = np.append(label_colorbar, label) 

        celltypes_matrix = celltypesDF.to_numpy()
        sns.heatmap(
            data=np.flip(celltypes_matrix[random_index]),
            xticklabels=False,
            yticklabels=False,
            ax=ax[2*i + len(factors)],
            cmap=list(choose_color_palette),
            )

        cbar = ax[2*i + len(factors)].collections[0].colorbar
        cbar.set_ticks(colorbar_numbers)
        cbar.set_ticklabels(label_colorbar)
        celltype_values = celltypes_matrix.ravel()

        for k in range(factors[1].shape[1]):
            ss = silhouette_samples(pps[ind, k].reshape(-1, 1), celltype_values) 
            for l, label in enumerate(np.unique(allcelltypes.index)):
                ss_score = np.mean(ss[celltype_values == l])
                silhouetteDF = pd.concat([silhouetteDF , pd.DataFrame({"Projs": i, "Silhoutte Score": ss_score, "Cell Type": [label], "Cmp.": [xticks[k]]})]) 
                
    sns.barplot(data=silhouetteDF.loc[silhouetteDF["Projs"] == 0], x="Cell Type", y = "Silhoutte Score", hue = "Cmp.", ax=ax[2*i + len(factors) + 1])
    sns.barplot(data=silhouetteDF.loc[silhouetteDF["Projs"] == 1], x="Cell Type", y = "Silhoutte Score", hue = "Cmp.", ax=ax[2*i + len(factors) + 2])


def reorder_table(projs):
    """Reorder a table's rows using heirarchical clustering"""
    assert projs.ndim == 2
    Z = sch.linkage(projs, method="centroid", optimal_ordering=True)
    index = sch.leaves_list(Z)
    return projs[index, :], index


def renamePlotSynthetic(xarray, ax):
    ax[2].set_title("Projection Matrix - " + "Time:0")
    ax[4].set_title("Projection Matrix - " + "Time:6")
    ax[6].set_title("Time:0")
    ax[7].set_title("Time:6")


def renamePlotIL2(ax):
    ax[2].set_yticklabels([f"Time:{i}" for i in [1, 2, 4]])
    ax[4].set_title("Projection Matrix - " + "Time:1")
    ax[6].set_title("Projection Matrix - " + "Time:2")
    ax[8].set_title("Time:1")
    ax[9].set_title("Time:2")
      
def renamePlotscRNA(ax):
    ax[2].set_title("Projection Matrix - " + "Acetylcysteine")
    ax[4].set_title("Projection Matrix - " + "Adapalene")
    ax[6].set_title("Acetylcysteine")
    ax[7].set_title("Adapalene")


def renamePlotsCoH(ax):
    ax[3].set_title("Projection Matrix - " + "Patient 0 - IFN")
    ax[5].set_title("Projection Matrix - " + "Patient 0 - IL10")
    ax[7].set_title("Projection Matrix - " + "Patient 0 - IL10")
    ax[8].set_title("Projection Matrix - " + "Patient 0 - IL2")