"""
CITEseq: Plotting genes per component
"""
from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotGeneral import plotGenePerCellType
import pandas as pd
import numpy as np
def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 12), (3, 4))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("factor_cache/CITEseq.h5ad", backed="r")
    XX = read_h5ad("/opt/pf2/CITE_Neighbors.h5ad", backed="r")
    X.obs["leiden"] = XX.obs["leiden"] 
    
    comps = [22, 33, 47, 48, 23, 31, 43]
    genes = top_bot_genes(X, cmp=comps[0])
    
    for i, gene in enumerate(genes):
        plotGenePerCellType(gene, X, ax[i], cellType="leiden")



    return f




def top_bot_genes(X, cmp, geneAmount=5):
    """Saves most pos/negatively genes"""
    df = pd.DataFrame(
        data=X.varm["Pf2_C"][:, cmp - 1], index=X.var_names, columns=["Component"]
    )
    df = df.reset_index(names="Gene")
    df = df.sort_values(by="Component")

    top = df.iloc[-geneAmount:, 0].values
    bot = df.iloc[:geneAmount, 0].values
    all_genes = np.concatenate([top, bot])
    
    return all_genes 