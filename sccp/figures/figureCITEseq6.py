"""
CITEseq: Cell type percentage per Leiden cluster per condition
"""

from anndata import read_h5ad
from .common import subplotLabel, getSetup
import seaborn as sns
from .commonFuncs.plotGeneral import cell_count_perc_df


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 3), (1, 1))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/pf2/CITEseq_fitted_annotated.h5ad", backed="r")

    df = cell_count_perc_df(X, celltype="leiden")

    sns.barplot(
        data=df,
        x="Cell Type",
        y="Cell Type Percentage",
        hue="Condition",
        ax=ax[0],
        errorbar=None,
    )

    return f
