"""
Figure 4a_c
"""

from anndata import read_h5ad

from ..factorization import correct_conditions
from .common import getSetup, subplotLabel
from .commonFuncs.plotLupus import plot_roc_fourthbatch
from .commonFuncs.plotPaCMAP import plot_labels_pacmap
import numpy as np
from scipy import sparse
import pandas as pd
import scanpy as sc
from .commonFuncs.plotGeneral import cell_count_perc_df, rotate_xaxis
from ..logisticReg import predaccuracy_lupus
from .commonFuncs.plotLupus import plot_accuracy_ranks_lupus
from .commonFuncs.plotGeneral import plot_r2x


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((9, 4), (1, 3))
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    
    # cell_comp_df = cell_count_perc_df(X, celltype="Cell Type")
    # cell_comp_df = cell_comp_df.pivot(index=["Condition", "Status", "Processing_Cohort", "condition_unique_idxs"], columns="Cell Type", values="Cell Type Percentage")
    # cell_comp_df = cell_comp_df.sort_values("condition_unique_idxs")
    
    # df = predaccuracy_lupus(cell_comp_df)
    # print(df)
    # y = [df.iloc[0], df.iloc[0]]
    # # # print(XXX)
    
    # cell_comp_gene_df = aggregate_anndata(X, celltype_col="Cell Type", condition_col="Condition", method="Average")
    # cell_comp_gene_df = cell_comp_gene_df.sort_values("condition_unique_idxs")
    
    # df = predaccuracy_lupus(cell_comp_gene_df)
    # # print(df)
    
    
    
    #  combined_df = cell_comp_df.merge(cell_comp_gene_df, on=["Condition", "Status", "Processing_Cohort", "condition_unique_idxs"], how="inner")
    # print(combined_df)
    
    # df = predaccuracy_lupus(combined_df)
    
    

    # ranks=[1, 2]
    # # prnt
    # plot_accuracy_ranks_lupus(X, ranks, ax[0], error_metric="roc_auc")
    # ax[0].set(xticks=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    # x = [0, 50]
    # y = [0.84, 0.84]
    # ax[0].plot(x, y, linestyle="--")

    # X.uns["Pf2_A"] = correct_conditions(X)
    # plot_roc_fourthbatch(X, ax[1])

    # plot_labels_pacmap(X, "Cell Type2", ax[2])

    return f

def aggregate_anndata(adata, celltype_col, condition_col, method="Average"):
    """Aggregate AnnData object by cell type and condition."""
    cell_types = adata.obs[celltype_col].unique()
    conditions = adata.obs[condition_col].unique()
    results = []

    for ct in cell_types:
        for cond in conditions:
            mask = (adata.obs[celltype_col] == ct) & (adata.obs[condition_col] == cond)
            group_data = adata[mask]
            if method == "Average":
                agg_values = np.mean(group_data.X, axis=0)
            elif method == "Sum":
                agg_values = (np.sum(group_data.X, axis=0)) / (np.shape(group_data.X)[0])

            sle_status = group_data.obs["SLE_status"].unique()[0]
            cohort = group_data.obs["Processing_Cohort"].unique()[0]
            idx = group_data.obs["condition_unique_idxs"].unique()[0]
            agg_values = np.ravel(agg_values)

    # Create a single result dictionary
            result_dict = {
                'Gene': adata.var_names,
                'Value': agg_values,
                'Cell Type': ct,
                'Condition': cond,
                'Status': sle_status,
                'Processing_Cohort': cohort,
                'condition_unique_idxs': idx
            }
            
            results.append(pd.DataFrame(result_dict))
    
    
    df = pd.concat(results, ignore_index=True)
    
    pivot_df = df.pivot_table(index=["Condition", "Status", "Processing_Cohort", "condition_unique_idxs"], columns=["Cell Type", "Gene"], values=["Value"])
    
    return pivot_df

