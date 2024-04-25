"""
Lupus: Two components weighted by condition factors 
"""

from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotLupus import samples_only_lupus
import seaborn as sns
import numpy as np
import pandas as pd
from ..factorization import correct_conditions
from matplotlib.axes import Axes
import anndata


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((5, 3), (1, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    
    plot_pair_cond_factors(X, cmp1=27, cmp2=28, ax=ax[0])
    plot_pair_cond_factors(X, cmp1=14, cmp2=4, ax=ax[1])


    return f



def plot_pair_cond_factors(X: anndata.AnnData, cmp1: int, cmp2: int, ax: Axes):
    """Plots two condition components weights"""
    cond_factors = correct_conditions(X)
    condStatus = samples_only_lupus(X)
    condStatus = condStatus.set_index("Condition")

    cmpWeights = np.concatenate(([cond_factors[:, cmp1-1]], [cond_factors[:, cmp2-1]]))
    df = pd.DataFrame(data=cmpWeights.transpose(), columns=[f"Cmp. {cmp1}", f"Cmp. {cmp2}"],
                       index=condStatus.index)

    df = df.merge(condStatus, left_index=True, right_index=True)

    sns.scatterplot(data=df, x=f"Cmp. {cmp1}", y=f"Cmp. {cmp2}", hue="SLE_status", ax=ax)