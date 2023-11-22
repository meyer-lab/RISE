from .common import getSetup
from ..gating import gateThomsonCells
from ..factorization import pf2
from .commonFuncs.plotFactors import (
    plotFactors,
)
from ..imports import import_thomson


def makeFigure():
    rank = 20
    data = import_thomson()
    gateThomsonCells(data)

    sampled_data = data[
        (data.obs["Cell Type"] != "T Cells") | (data.obs["Condition"] != "CTRL4")
    ]

    ax, f = getSetup((12, 12), (2, 3))

    origX = pf2(data, rank, doEmbedding=False)

    plotFactors(origX, ax[0:3], reorder=(0, 2))

    # sampled_data = data

    sampledX = pf2(sampled_data, rank, doEmbedding=False)

    plotFactors(sampledX, ax[3:6], reorder=(0, 2))

    return f
