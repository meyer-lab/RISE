import numpy as np
import scanpy as sc
import anndata
from .common import subplotLabel, getSetup
from ..factorization import pf2
from ..imports import import_thomson
from matplotlib.axes import Axes
from tlviz.factor_tools import factor_match_score as fms
from tensorly.cp_tensor import CPTensor
import seaborn as sns


def calculateFMS(A: anndata.AnnData, B: anndata.AnnData) -> float:
    factors = [A.uns["Pf2_A"], A.uns["Pf2_B"], A.varm["Pf2_C"]]
    A_CP = CPTensor(
        (
            A.uns["Pf2_weights"],
            factors,
        )
    )

    factors = [B.uns["Pf2_A"], B.uns["Pf2_B"], B.varm["Pf2_C"]]
    B_CP = CPTensor(
        (
            B.uns["Pf2_weights"],
            factors,
        )
    )

    return fms(A_CP, B_CP, consider_weights=False, skip_mode=1)  # type: ignore


def plotFMSpercentDrop(
    X: anndata.AnnData,
    ax: Axes,
    percentList: np.ndarray,
    runs: int,
    rank=20,
):
    # pf2 on original dataset
    dataX = pf2(X, rank, doEmbedding=False)

    fmsLists = []

    # loop to do multiple runs
    for j in range(0, runs, 1):
        scores = [1.0]

        # loop to compare sampled dataset to original
        for i in percentList[1:]:
            sampled_data: anndata.AnnData = sc.pp.subsample(X, fraction=1 - (i / 100), copy=True)  # type: ignore
            sampledX = pf2(sampled_data, rank, doEmbedding=False)

            fmsScore = calculateFMS(dataX, sampledX)
            scores.append(fmsScore)

        fmsLists.append(scores)

    random_colors = sns.color_palette("husl", len(fmsLists))
    for n in range(0, runs, 1):
        ax.plot(percentList, fmsLists[n], label=f"Run {n+1}", color=random_colors[n])

    # percent dropped vs fms graph
    ax.set_xlabel("Percentage of Data Dropped")
    ax.set_ylabel("FMS")
    ax.set_title("Percent of Data Dropped vs FMS")
    ax.legend()


def resample(data: anndata.AnnData) -> anndata.AnnData:
    indices = np.random.randint(0, data.shape[0], size=(data.shape[0],))
    data = data[indices].copy()
    return data


def plotRankTest(
    X: anndata.AnnData,
    ax: Axes,
    ranksList: list[int],
):
    fmsList = []

    # testing different ranks input into function with one percent valued dropped
    for i in ranksList:
        dataX = pf2(X, rank=i, doEmbedding=False)

        sampledX = pf2(resample(X), rank=i, doEmbedding=False)

        fmsScore = calculateFMS(dataX, sampledX)
        fmsList.append(fmsScore)

    # rank vs fms graph
    ax.plot(ranksList, fmsList, color="pink")
    ax.set_xlabel("Rank")
    ax.set_ylabel("FMS")
    ax.set_title("Rank vs FMS")


# testing functions
def makeFigure():
    X = import_thomson()

    ax, f = getSetup((15, 5), (1, 3))

    subplotLabel(ax)

    percentList = np.arange(0.0, 11.0, 2.0)
    plotFMSpercentDrop(X, ax[0], percentList=percentList, runs=3)

    ranks = list(range(1, 25, 2))
    plotRankTest(X, ax[2], ranksList=ranks)

    return f
