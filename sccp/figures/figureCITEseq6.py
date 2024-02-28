"""
CITEseq: Plotting cell count per Leiden cluster per condition
"""
from anndata import read_h5ad
from .common import subplotLabel, getSetup
import numpy as np
import seaborn as sns

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 2), (1, 1))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/pf2/CITEseq_fitted_annotated.h5ad", backed="r")
    
    plotCellCount(X, ax[0])

    return f

def plotCellCount(X, ax, celltype="leiden"):
    """Plots cell count per cluster per condition and as a percentage"""
    df = X.obs[[celltype, "Condition"]].reset_index(drop=True)
    sns.histplot(data=df, x=celltype, hue="Condition", ax=ax, multiple="dodge", shrink=.7)
