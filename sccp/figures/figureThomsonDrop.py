from .common import getSetup
from ..gating import gateThomsonCells
from ..factorization import pf2
from .commonFuncs.plotFactors import (
    plotConditionsFactors,
    plotCellState,
    plotGeneFactors,
)
from ..imports import import_thomson
from .figureThomson1 import groupDrugs


def makeFigure():
    rank = 20
    data = import_thomson()
    gateThomsonCells(data)

    sampled_data = data[
        (data.obs["Cell Type"] != "T Cells") | (data.obs["Condition"] != "CTRL4")
    ]

    ax, f = getSetup((12, 12), (2, 3))

    origX = pf2(data, rank, doEmbedding=False)

    plotConditionsFactors(origX, ax[0], groupDrugs(origX.obs["Condition"]), ThomsonNorm=True)
    plotCellState(origX, ax[1])
    plotGeneFactors(origX, ax[2])

    # sampled_data = data

    sampledX = pf2(sampled_data, rank, doEmbedding=False)
    
    plotConditionsFactors(sampledX, ax[3], groupDrugs(origX.obs["Condition"]), ThomsonNorm=True)
    plotCellState(sampledX, ax[4])
    plotGeneFactors(sampledX, ax[5])


    return f
