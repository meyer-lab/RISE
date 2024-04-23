"""
Thomson dataset: Cell counts and cell type percentages per condition.
"""

from anndata import read_h5ad
import seaborn as sns
from .common import subplotLabel, getSetup
from .commonFuncs.plotGeneral import cell_count_perc_df, rotate_xaxis


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((11, 14), (4, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/pf2/thomson_fitted.h5ad", backed="r")
    
    plot_cell_count(X, ax[0])
    
    df = cell_count_perc_df(X, celltype="Cell Type")
    for i, (name, group) in enumerate(df.groupby("Cell Type", observed=True)):
        sns.barplot(
            data=group,
            x="Condition",
            y="Cell Type Percentage",
            color="k",
            ax=ax[i + 1],
        )
        rotate_xaxis(ax[i + 1])
        ax[i + 1].set(ylabel=f"{name} Percentage")

    return f


def plot_cell_count(X, ax):
    """Plots overall cell count for Chen et al."""
    df = X.obs[["Condition"]].reset_index(drop=True)
    dfCond = df.groupby(["Condition"], observed=True).size().reset_index(name="Cell Count")

    sns.barplot(data=dfCond, x="Condition", y="Cell Count", color="k", ax=ax)
    rotate_xaxis(ax)