"""Figure S4: PCA and Pf2 PaCMAP labeled by genes and drugsFMS removing percentages of dataset and FMS across different components"""

import numpy as np
import scanpy as sc
import anndata
from matplotlib.axes import Axes
from tlviz.factor_tools import factor_match_score as fms
from tensorly.cp_tensor import CPTensor
import seaborn as sns
import pandas as pd
from .common import subplotLabel, getSetup
from ..factorization import pf2
# from ..imports import import_thomson


def makeFigure():
    ax, f = getSetup((6, 3), (1, 2))
    subplotLabel(ax)

    # X = import_thomson()
    # percentList = np.arange(0.0, 8.0, 5.0)
    # plot_fms_percent_drop(X, ax[0], percentList=percentList, runs=3)

    # ranks = list(range(1, 3))
    # plot_fms_diff_ranks(X, ax[1], ranksList=ranks, runs=3)

    return f


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


def plot_fms_percent_drop(
    X: anndata.AnnData,
    ax: Axes,
    percentList: np.ndarray,
    runs: int,
    rank=20,
):
    dataX = pf2(X, rank, doEmbedding=False)

    fmsLists = []

    for j in range(0, runs, 1):
        scores = [1.0]

        for i in percentList[1:]:
            sampled_data: anndata.AnnData = sc.pp.subsample(
                X, fraction=1 - (i / 100), random_state=j, copy=True
            )  # type: ignore
            sampledX = pf2(sampled_data, rank, random_state=j + 2, doEmbedding=False)

            fmsScore = calculateFMS(dataX, sampledX)
            scores.append(fmsScore)

        fmsLists.append(scores)

    runsList_df = []
    for i in range(0, runs):
        for j in range(0, len(percentList)):
            runsList_df.append(i)
    percentList_df = []
    for i in range(0, runs):
        for j in range(0, len(percentList)):
            percentList_df.append(percentList[j])
    fmsList_df = []
    for sublist in fmsLists:
        fmsList_df += sublist
    df = pd.DataFrame(
        {
            "Run": runsList_df,
            "Percentage of Data Dropped": percentList_df,
            "FMS": fmsList_df,
        }
    )

    sns.lineplot(data=df, x="Percentage of Data Dropped", y="FMS", ax=ax)
    ax.set_ylim(0, 1)


def resample(data: anndata.AnnData) -> anndata.AnnData:
    indices = np.random.randint(0, data.shape[0], size=(data.shape[0],))
    data = data[indices].copy()
    return data


def plot_fms_diff_ranks(
    X: anndata.AnnData,
    ax: Axes,
    ranksList: list[int],
    runs: int,
):
    fmsLists = []

    for j in range(0, runs, 1):
        scores = []
        for i in ranksList:
            dataX = pf2(X, rank=i, random_state=j, doEmbedding=False)

            sampledX = pf2(resample(X), rank=i, random_state=j, doEmbedding=False)

            fmsScore = calculateFMS(dataX, sampledX)
            scores.append(fmsScore)
        fmsLists.append(scores)

    runsList_df = []
    for i in range(0, runs):
        for j in range(0, len(ranksList)):
            runsList_df.append(i)
    ranksList_df = []
    for i in range(0, runs):
        for j in range(0, len(ranksList)):
            ranksList_df.append(ranksList[j])
    fmsList_df = []
    for sublist in fmsLists:
        fmsList_df += sublist
    df = pd.DataFrame(
        {"Run": runsList_df, "Component": ranksList_df, "FMS": fmsList_df}
    )

    sns.lineplot(data=df, x="Component", y="FMS", ax=ax)
    ax.set_ylim(0, 1)
