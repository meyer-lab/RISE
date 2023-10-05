"""
This file contains functions that are used in multiple figures.
"""
from string import ascii_letters
import sys
import time
import seaborn as sns
import matplotlib
from matplotlib.figure import Figure
from matplotlib import gridspec, pyplot as plt
import numpy as np
import pickle
import pandas as pd
from .commonFuncs.plotFactors import reorder_table


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


def getSetup(figsize, gridd) -> tuple[list, Figure]:
    """Establish figure set-up with subplots."""
    sns.set(
        style="whitegrid",
        font_scale=0.7,
        color_codes=True,
        palette="colorblind",
        rc={"grid.linestyle": "dotted", "axes.linewidth": 0.6},
    )

    # Setup plotting space and grid
    f = plt.figure(figsize=figsize, layout="constrained")
    gs1 = gridspec.GridSpec(gridd[0], gridd[1], figure=f)

    # Get list of axis objects
    ax = [f.add_subplot(gs1[x]) for x in range(gridd[0] * gridd[1])]

    return ax, f


def subplotLabel(axs):
    """Place subplot labels on figure."""
    for ii, ax in enumerate(axs):
        ax.text(
            -0.2,
            1.2,
            ascii_letters[ii],
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
        )


def genFigure():
    """Main figure generation function."""
    start = time.time()
    nameOut = "figure" + sys.argv[1]

    exec(f"from sccp.figures.{nameOut} import makeFigure", globals())
    ff = makeFigure()
    ff.savefig(f"./output/{nameOut}.svg", dpi=300, bbox_inches="tight", pad_inches=0)
    ff.savefig(f"./output/{nameOut}.png", dpi=300, bbox_inches="tight", pad_inches=0)

    print(f"Figure {sys.argv[1]} is done after {time.time() - start} seconds.\n")


def savePf2(weight, factors, projs, dataName: str):
    """Saves weight factors and projections for one dataset for a component"""
    rank = len(weight)
    np.save(f"./sccp/data/{dataName}/{dataName}_WeightCmp{rank}.npy", weight)

    factor = ["A", "B", "C"]
    for i in range(3):
        np.save(
            f"./sccp/data/{dataName}/{dataName}_Factor{factor[i]}Cmp{rank}.npy",
            factors[i],
        )

    np.save(
        f"./sccp/data/{dataName}/{dataName}_ProjCmp{rank}.npy",
        np.concatenate(projs, axis=0),
    )


def openPf2(rank: int, dataName: str, optProjs=False):
    """Opens weight factors and projections for one dataset for a component as numpy arrays"""
    weight = np.load(
        f"./sccp/data/{dataName}/{dataName}_WeightCmp{rank}.npy",
        allow_pickle=True,
    )
    factors = [
        np.load(
            f"./sccp/data/{dataName}/{dataName}_FactorACmp{rank}.npy",
            allow_pickle=True,
        ),
        np.load(
            f"./sccp/data/{dataName}/{dataName}_FactorBCmp{rank}.npy", allow_pickle=True
        ),
        np.load(
            f"./sccp/data/{dataName}/{dataName}_FactorCCmp{rank}.npy",
            allow_pickle=True,
        ),
    ]

    if optProjs is False:
        projs = np.load(
            f"./sccp/data/{dataName}/{dataName}_ProjCmp{rank}.npy",
            allow_pickle=True,
        )
    else:
         projs = np.load(
            f"/opt/andrew/{dataName}/{dataName}_ProjCmp"+str(rank)+".npy", 
            allow_pickle=True)
        

    return weight, factors, projs


def saveUMAP(fit_points, rank: int, dataName: str):
    """Saves UMAP points locally, large files uploaded manually to opt"""
    f_name = f"./sccp/data/{dataName}/{dataName}_UMAPCmp{rank}.sav"
    pickle.dump(fit_points, open(f_name, "wb"))


def openUMAP(rank: int, dataName: str, opt=True):
    """Opens UMAP points for plotting, defaults to using the opt folder (for big files)"""
    if opt == True:
        f_name = f"/opt/andrew/{dataName}/{dataName}_UMAPCmp{rank}.sav"
    else:
        f_name = f"./sccp/data/{dataName}/{dataName}_UMAPCmp{rank}.sav"

    return pickle.load((open(f_name, "rb")))


def flattenData(data):
    """Flattens tensor into dataframe"""
    cellCount = []
    for i in range(len(data.X_list)):
        cellCount = np.append(cellCount, data.X_list[i].shape[0])

    condNames = []

    for i in range(len(data.X_list)):
        condNames = np.append(
            condNames, np.repeat(data.condition_labels[i], cellCount[i])
        )
    flatData = np.concatenate(data.X_list, axis=0)
    dataDF = pd.DataFrame(data=flatData, columns=data.variable_labels)
    dataDF["Condition"] = condNames

    return dataDF


def flattenWeightedProjs(data, factors, projs):
    """Flattens tensor into dataframe"""
    cellCount = []
    for i in range(len(data.X_list)):
        cellCount = np.append(cellCount, data.X_list[i].shape[0])

    condNames = []

    for i in range(len(data.X_list)):
        condNames = np.append(
            condNames, np.repeat(data.condition_labels[i], cellCount[i])
        )

    weightedProjs = projs @ factors[1]

    weightedProjs = weightedProjs / np.max(np.abs(weightedProjs))

    cmpNames = [f"Cmp. {i}" for i in np.arange(1, weightedProjs.shape[1] + 1)]
    dataDF = pd.DataFrame(data=weightedProjs, columns=cmpNames)
    dataDF["Condition"] = condNames

    return dataDF


def saveGeneFactors(factors, data, dataName):
    """Saves genes factors based on weight."""
    rank = factors[0].shape[1]
    yt = data.variable_labels
    X = factors[2]

    max_weight = np.max(np.abs(X), axis=1)
    kept_idxs = max_weight > 0.08
    X = X[kept_idxs]
    yt = yt[kept_idxs]

    X, ind = reorder_table(X)
    yt = yt[ind]

    X = X / np.max(np.abs(X))

    if len(yt) > 40:
        df = pd.DataFrame(
            data=X, index=yt, columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)]
        )
        df.to_csv(f"/opt/andrew/{dataName}/{dataName}TopBotGenes_Cmp{rank}.csv")

        geneAmount = 20
        genesTop = np.empty((geneAmount, X.shape[1]), dtype="<U10")
        genesBottom = np.empty((geneAmount, X.shape[1]), dtype="<U10")
        sort_idx = np.argsort(X, axis=0)

        for j in range(rank):
            sortGenes = yt[sort_idx[:, j]]
            genesTop[:, j] = np.flip(sortGenes[-geneAmount:])
            genesBottom[:, j] = sortGenes[:geneAmount]

        dfTop = pd.DataFrame(
            data=genesTop, columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)]
        )
        dfBot = pd.DataFrame(
            data=genesBottom, columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)]
        )

        dfTop.to_csv(f"/opt/andrew/{dataName}/{dataName}TopGenes_Cmp{rank}.csv")
        dfBot.to_csv(f"/opt/andrew/{dataName}/{dataName}BotGenes_Cmp{rank}.csv")


def repeatLabels(condLabels, data, dataDF):
    """Repeats a label in original AnnData file"""
    cellCount = dataDF.groupby(["Condition"]).size().values
    return [np.repeat(condLabels[i], cellCount[i]) for i in range(len(data.X_list))]
