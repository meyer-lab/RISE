"""
Lupus: UMAP labeled by cell type
"""

from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotUMAP import plotLabelsUMAP
import numpy as np
import seaborn as sns
import pandas as pd

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 6), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad", backed="r")

    plotLabelsUMAP(X, "Cell Type", ax[0])
    plotLabelsUMAP(X, "Cell Type2", ax[1])
    
    plot_pair_gene_factors(X, 22, 28, ax[2])
    

    return f


def plot_pair_gene_factors(X, cmp1, cmp2, ax):
    """Plots two gene components weights"""
    cmpWeights = np.concatenate(([X.varm["Pf2_C"][:, cmp1-1]], [X.varm["Pf2_C"][:, cmp2-1]]))
    df = pd.DataFrame(data=cmpWeights.transpose(), columns=[f"Cmp. {cmp1}", f"Cmp. {cmp2}"])
    sns.scatterplot(data=df, x=f"Cmp. {cmp1}", y=f"Cmp. {cmp2}", ax=ax)
    
