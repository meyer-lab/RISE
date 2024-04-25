"""
Lupus: PaCMAP labeled by cell type and gene factor weights for 2 components
"""

from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotPaCMAP import plot_labels_pacmap
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.axes import Axes
import anndata


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 6), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad", backed="r")

    plot_labels_pacmap(X, "Cell Type", ax[0])
    plot_labels_pacmap(X, "Cell Type2", ax[1])
    
    plot_pair_gene_factors(X, 22, 28, ax[2])
    

    return f


def plot_pair_gene_factors(X: anndata.AnnData, cmp1: int, cmp2: int, ax: Axes):
    """Plots two gene components weights"""
    cmpWeights = np.concatenate(([X.varm["Pf2_C"][:, cmp1-1]], [X.varm["Pf2_C"][:, cmp2-1]]))
    df = pd.DataFrame(data=cmpWeights.transpose(), columns=[f"Cmp. {cmp1}", f"Cmp. {cmp2}"])
    sns.scatterplot(data=df, x=f"Cmp. {cmp1}", y=f"Cmp. {cmp2}", ax=ax)
    
