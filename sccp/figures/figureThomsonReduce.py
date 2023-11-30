"""Downsample Thomson and see how much the factors change.
The goal here is to see how much the dataset size matters."""
import scanpy as sc
from .common import subplotLabel, getSetup
from ..factorization import pf2
from .commonFuncs.plotFactors import (
    plotConditionsFactors,
    plotCellState,
    plotGeneFactors,
)
from ..imports import import_thomson
from tlviz.factor_tools import factor_match_score as fms
from tensorly.cp_tensor import CPTensor
from .figureThomson1 import groupDrugs


def makeFigure():
    rank = 20
    X = import_thomson()

    ax, f = getSetup((12, 12), (2, 3))
    subplotLabel(ax)

    # figures for all dataset
    dataX = pf2(X, rank, random_state=1)

    factors = [dataX.uns["Pf2_A"], dataX.uns["Pf2_B"], dataX.varm["Pf2_C"]]
    plotConditionsFactors(dataX, ax[0], groupDrugs(dataX.obs["Condition"]), ThomsonNorm=True)
    plotCellState(dataX, ax[1])
    plotGeneFactors(dataX, ax[2])
    dataXcp = CPTensor(
        (
            dataX.uns["Pf2_weights"],
            factors,
        )
    )

    sampled_data = sc.pp.subsample(X, fraction=0.99, random_state=0, copy=True)
    sampledX = pf2(sampled_data, rank, random_state=3)  # type: ignore
    sampled_factors = [
        sampledX.uns["Pf2_A"],
        sampledX.uns["Pf2_B"],
        sampledX.varm["Pf2_C"],
    ]
    plotConditionsFactors(sampledX, ax[3], groupDrugs(dataX.obs["Condition"]), ThomsonNorm=True)
    plotCellState(sampledX, ax[4])
    plotGeneFactors(sampledX, ax[5])
    sampledXcp = CPTensor(
        (
            sampledX.uns["Pf2_weights"],
            sampled_factors,
        )
    )
    
    # factor score match
    fmsScore = fms(dataXcp, sampledXcp, consider_weights=True, skip_mode=None)
    print(fmsScore)

    return f
