import anndata
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.metrics import RocCurveDisplay
from ...logisticReg import predaccuracy_lupus
from ..commonFuncs.plotGeneral import cell_count_perc_lupus_df
from ...logisticReg import (
    predaccuracy_ranks_lupus,
    roc_lupus_fourtbatch,
)


def samples_only_lupus(X: anndata.AnnData):
    """Obtain samples once only with corresponding observations"""
    samples = X.obs
    df_samples = samples.drop_duplicates(subset="condition_unique_idxs")
    df_samples = df_samples.sort_values("condition_unique_idxs")

    return df_samples


def plot_accuracy_ranks_lupus(
    X: anndata.AnnData, ranks: np.ndarray, ax: Axes, error_metric="roc_auc", bootstrap: bool = False
):
    """Plots results from Pf2 test of various ranks using defined error metric
    and logistic reg"""
    pred_accuracy_df = predaccuracy_ranks_lupus(
        X, samples_only_lupus(X), ranks, error_metric, bootstrap
    )

    sns.lineplot(
        data=pred_accuracy_df,
        x="Component",
        y=error_metric,
        ax=ax,
    )
    if bootstrap is False:
        sns.scatterplot(
            data=pred_accuracy_df,
            x="Component",
            y=error_metric,
            ax=ax,
        )
    ax.set(ylim=[-0.05, 1.05])


def plot_roc_fourthbatch(X: anndata.AnnData, ax: Axes):
    """Plots ROC curve for prediction"""
    y_test, sle_decisions = roc_lupus_fourtbatch(X, samples_only_lupus(X))

    RocCurveDisplay.from_predictions(
        y_test, sle_decisions, pos_label=True, plot_chance_level=True, ax=ax
    )


def plot_all_bulk_pred(X, ax, accuracy_metric="accuracy"):
    cell_comp_df = cell_count_perc_lupus_df(X, celltype="Cell Type")
    cell_comp_df = cell_comp_df.pivot(index=["Condition", "Status", "Processing_Cohort", "condition_unique_idxs"], columns="Cell Type", values="Cell Type Percentage")
    cell_comp_df = cell_comp_df.sort_values("condition_unique_idxs")
    
    cell_comp_pred = predaccuracy_lupus(cell_comp_df, error_metric=accuracy_metric)
    y_cell_comp = [cell_comp_pred.iloc[0], cell_comp_pred.iloc[0]]
    
    cell_comp_gene_df = aggregate_anndata(X, celltype_col="Cell Type", condition_col="Condition", method="Average")
    cell_comp_gene_df = cell_comp_gene_df.sort_values("condition_unique_idxs")
    cell_comp_gene_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in cell_comp_gene_df.columns]
    
    cell_comp_gene_pred = predaccuracy_lupus(cell_comp_gene_df, error_metric=accuracy_metric)
    y_cell_comp_gene = [cell_comp_gene_pred.iloc[0], cell_comp_gene_pred.iloc[0]]
    
    combined_df = pd.concat([cell_comp_df.reset_index(), cell_comp_gene_df.reset_index(drop=True)], axis=1)
    combined_df = combined_df.set_index(["Condition", "Status", "Processing_Cohort", "condition_unique_idxs"])
    
    combined_pred = predaccuracy_lupus(combined_df, error_metric=accuracy_metric)
    y_combined = [combined_pred.iloc[0], combined_pred.iloc[0]]
    
    x = [0, 50]
    y_perez_et_al = [0.84, 0.84]
    if accuracy_metric == "roc_auc":
        ax.plot(x, y_perez_et_al, linestyle="--", color="k")
    ax.plot(x, y_cell_comp, linestyle="--", color="r")
    ax.plot(x, y_cell_comp_gene, linestyle="--", color="g")
    ax.plot(x, y_combined, linestyle="--", color="m")
    

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

