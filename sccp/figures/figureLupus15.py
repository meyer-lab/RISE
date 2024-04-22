"""
Lupus: Plots of amount of cells and cell type distribution across all experiments
"""

from anndata import read_h5ad
import seaborn as sns
from .common import subplotLabel, getSetup
from .commonFuncs.plotGeneral import cell_count_perc_df, rotate_xaxis


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 6), (4, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad", backed="r")

    celltype = ["Cell Type", "Cell Type2", "leiden"]
    label = ["Cell Type Percentage", "Cell Count"]
    plot = 0
    
    for i in range(len(celltype)):
        for j in range(len(label)):
            df = cell_count_perc_df(X, celltype=celltype[i], status=True)
            sns.boxplot(data=df, x="Cell Type", y=label[j], 
                        hue="SLE_status", showfliers=False, ax=ax[plot])
            rotate_xaxis(ax[plot])
            plot+=1

    plot_cell_count_status(X, ax[6])
    f.delaxes(ax[7])   

    return f

def plot_cell_count_status(X, ax):
    """Plots overall cell count for SLE and healthy patients"""
    df = X.obs[["SLE_status", "Condition"]].reset_index(drop=True)
    dfCond = df.groupby(["Condition","SLE_status"], observed=True).size().reset_index(name="Cell Count")

    sns.boxplot(data=dfCond, x="SLE_status", y="Cell Count", hue="SLE_status", showfliers=False, ax=ax)


    return 