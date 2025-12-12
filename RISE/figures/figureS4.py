"""
Figure S4
"""

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.axes import Axes
from tensorly.cp_tensor import CPTensor
from tlviz.factor_tools import factor_match_score as fms

from ..factorization import pf2
from .common import getSetup, subplotLabel

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


def calculateFMS(A: anndata.AnnData, B: anndata.AnnData):
    """Calculate Factor Match Score (FMS) between two RISE decompositions.

    Factor Match Score measures the similarity between two tensor decompositions
    by comparing their factor matrices. Values range from 0 (no similarity) to 1
    (identical factors). Used to assess decomposition stability across different
    initializations or data subsamples.

    Parameters
    ----------
    A : anndata.AnnData
        First AnnData object with RISE decomposition results. Must contain:
        - A.uns["Pf2_weights"]: Component weights
        - A.uns["Pf2_A"]: Condition factors
        - A.uns["Pf2_B"]: Eigen-state factors
        - A.varm["Pf2_C"]: Gene factors
    B : anndata.AnnData
        Second AnnData object with RISE decomposition results. Must have the
        same rank as A and contain the same decomposition attributes.

    Returns
    -------
    float
        Factor Match Score between 0 and 1. Higher values indicate more similar
        decompositions. Typically: >0.9 = highly stable, >0.6 = acceptable,
        <0.6 = unstable decomposition.

    Notes
    -----
    This function uses tlviz.factor_tools.factor_match_score with weights
    consideration disabled (consider_weights=False) and skipping the condition
    mode (skip_mode=1) for stability assessment across replicate decompositions.
    """
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
    rank: int = 20,
):
    """Plots FMS score when percentage is removed from data"""
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
        for _j in range(0, len(percentList)):
            runsList_df.append(i)
    percentList_df = []
    for _i in range(0, runs):
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
    """Bootstrapping dataset"""
    indices = np.random.randint(0, data.shape[0], size=(data.shape[0],))
    data = data[indices].copy()
    return data


def plot_fms_diff_ranks(
    X: anndata.AnnData,
    ax: Axes,
    ranksList: list[int],
    runs: int,
):
    """Plot Factor Match Score (FMS) across different ranks to assess stability.

    FMS measures the reproducibility of PARAFAC2 decomposition results across
    multiple runs. Values above ~0.6 indicate stable, reproducible components.
    This helps determine which ranks produce reliable decompositions that are
    not overly sensitive to initialization or noise.

    Parameters
    ----------
    X : anndata.AnnData
        Preprocessed AnnData object containing single-cell RNA-seq data.
        Must have X.obs["condition_unique_idxs"] for RISE decomposition.
    ax : matplotlib.axes.Axes
        Matplotlib axes object to plot on.
    ranksList : list of int
        List of rank values to test (e.g., [1, 5, 10, 15, 20, 25, 30]).
        Each rank will be run multiple times to compute FMS.
    runs : int
        Number of independent runs per rank to use for FMS calculation.
        Higher values give more reliable FMS estimates but take longer.
        Typical values: 3-5 runs.

    Notes
    -----
    FMS values interpretation:
    - FMS > 0.9: Highly stable decomposition
    - FMS > 0.6: Acceptably stable decomposition
    - FMS < 0.6: Unstable, consider lower rank or more data
    """
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
        for _j in range(0, len(ranksList)):
            runsList_df.append(i)
    ranksList_df = []
    for _i in range(0, runs):
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
