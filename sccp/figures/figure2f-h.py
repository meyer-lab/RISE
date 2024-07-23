"""
Thomson: Plotting Pf2 factors and weights
"""

from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_gene_factors,
    plot_factor_weight,
)
import numpy as np
import pandas as pd
import anndata


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((10, 12), (2, 2))
    subplotLabel(ax)

    X = read_h5ad("/opt/pf2/thomson_fitted.h5ad", backed="r")

    drugNames = groupDrugs(X, "Condition")

    plot_condition_factors(X, ax[0], drugNames, ThomsonNorm=True, groupConditions=True)
    plot_eigenstate_factors(X, ax[1])
    plot_gene_factors(X, ax[2])

    return f


def groupDrugs(X: anndata, label_name: str):
    """Groups drugs of similar category"""
    labels = X.obs[label_name]
    names = np.unique(labels)

    glucs = [
        "Triamcinolone Acetonide",
        "Loteprednol etabonate",
        "Betamethasone Valerate",
        "Budesonide",
        "Meprednisone",
    ]
    for i in glucs:
        names[names == i] = "Glucocorticoids"

    ctrl = ["CTRL1", "CTRL2", "CTRL3", "CTRL4", "CTRL5", "CTRL6"]
    for i in ctrl:
        names[names == i] = "Control"

    names[names == "Everolimus (RAD001)"] = "mTOR Inhibitor"
    names[names == "Rapamycin (Sirolimus)"] = "mTOR Inhibitor"
    names[names == "Alprostadil"] = "Prostaglandin"
    names[names == "Cyclosporine"] = "Calcineruin Inhibitor"

    condition = [
        "Glucocorticoids",
        "Control",
        "Prostaglandin",
        "mTOR Inhibitor",
        "Calcineruin Inhibitor",
    ]

    names = pd.Series([c if c in condition else "Other" for c in names])

    return names
