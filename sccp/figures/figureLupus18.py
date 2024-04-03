"""
Thomson: Plotting normalized genes and separating data by status (and celltype)
"""
from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
import pandas as pd


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((25, 20), (5, 6))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    
    
    comps = [22, 28]

    total_df = pd.DataFrame([])

    cmpWeights = np.concatenate(([X.varm["Pf2_C"][:, comps[0]-1]], [X.varm["Pf2_C"][:, comps[1]-1]]))

    df = pd.DataFrame(data=cmpWeights.transpose(), index=X.var_names, columns=["Component 22", "Component 28"])

    sns.scatterplot(data=df, x="Component 22", y="Component 28", ax=ax[0])

    sns.scatterplot(data=df, x="Component 22", y="Component 28", ax=ax[1])
    # ax[1].set(xscale="symlog", yscale="symlog", xlim=[-.5, .1])
    # ax[1].set_yticks([.1 , .01, .001, 0, -.001, .01, .1])
    
    ax[1].set(xlim=[-.5, .5],yscale="symlog",xscale="symlog" )
    ax[1].set_yticks([1, .1 , .01, .001, 0, -.001, .01, .1])



    return f




