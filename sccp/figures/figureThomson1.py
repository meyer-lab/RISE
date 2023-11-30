"""
Thomson: Plotting Pf2 factors and weights
"""
from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotFactors import (
    plotConditionsFactors,
    plotCellState,
    plotGeneFactors,
    plotWeight,
)
import numpy as np
import pandas as pd


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 12), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("factor_cache/Thomson.h5ad", backed="r")

    drugNames = groupDrugs(X.obs["Condition"])

    plotConditionsFactors(X, ax[0], drugNames, ThomsonNorm=True)
    plotCellState(X, ax[1])
    plotGeneFactors(X, ax[2])
    plotWeight(X, ax[3])

    return f


def groupDrugs(labels):
    """Groups drugs of similar category"""
    names = np.unique(labels)

    glucs = [
        "Triamcinolone Acetonide",
        "Loteprednol etabonate",
        "Betamethasone Valerate",
        "Budesonide",
        "Meprednisone",
    ]
    for i in glucs:
        names[names == i] = "Glucocoritcoids"

    ctrl = ["CTRL1", "CTRL2", "CTRL3", "CTRL4", "CTRL5", "CTRL6"]
    for i in ctrl:
        names[names == i] = "Control"

    names[names == "Everolimus (RAD001)"] = "mTOR Inhibitor"
    names[names == "Rapamycin (Sirolimus)"] = "mTOR Inhibitor"
    names[names == "Alprostadil"] = "Prostaglandin"
    names[names == "Cyclosporine"] = "Calcineruin Inhibitor"

    condition = [
        "Glucocoritcoids",
        "Control",
        "Prostaglandin",
        "mTOR Inhibitor",
        "Calcineruin Inhibitor",
    ]

    names = pd.Series([c if c in condition else "Other" for c in names])

    return names
