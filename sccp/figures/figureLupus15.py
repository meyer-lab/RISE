"""
Investigation of raw data for Thomson dataset
"""
from .common import subplotLabel, getSetup, flattenData
import seaborn as sns
import numpy as np
import numpy as np
from ..imports.scRNA import load_lupus_data

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((10, 8), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    lupus_tensor, obs = load_lupus_data()
    dataDF = flattenData(lupus_tensor)

    cell_types = obs[["cell_type_broad", "SLE_status"]].reset_index(drop=True)
    dataDF["Cell Type"] = cell_types["cell_type_broad"].values
    
    dfCond = dataDF.groupby(["Condition"]).size().reset_index(name="Cell Number") 
    sns.histplot(data=dfCond, x="Cell Number", bins=15, color="k", ax=ax[0])
    ax[0].set(ylabel="# of Experiments")
    
    dfCellType = dataDF.groupby(["Cell Type", "Condition"]).size().reset_index(name="Count") 
    for i, cond in enumerate(lupus_tensor.condition_labels):
        dfCellType.loc[dfCellType["Condition"] == cond, "Count"] = 100 * dfCellType.loc[dfCellType["Condition"] == cond, "Count"].to_numpy()/dfCond.loc[dfCond["Condition"] == cond]["Cell Number"].to_numpy()
    
    dfCellType.rename(columns={"Count": "Cell Type Percentage"}, inplace=True)
    for i, celltype in enumerate(np.unique(dfCellType["Cell Type"])):
        sns.histplot(data=dfCellType.loc[dfCellType["Cell Type"] == celltype], x="Cell Type Percentage", bins=15, color="k", ax=ax[i+1])
        ax[i+1].set(title=celltype, ylabel="# of Experiments")

    return f
