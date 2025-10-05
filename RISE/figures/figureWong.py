"""
Figure Wong Generation Script

This module generates mean expression data for CM cells across all genes.
"""

import anndata
import numpy as np
import pandas as pd
from .common import getSetup, subplotLabel
from ..imports import import_lupus


def makeFigure():
    ax, f = getSetup((22, 9), (4, 10))
    subplotLabel(ax)
    
    X = import_lupus(geneThreshold=0)
    print("Using import_lupus data:")
    print(X)
    print("Available cell types in dataset:")
    print(X.obs["Cell Type"].value_counts())
    
    # Get the final dataframe with mean gene expression for each sample 
    # for CM population across all genes
    df_final = get_mean_expression_cm_cells(X)
    print(f"Final dataframe shape: {df_final.shape}")
    print(df_final.head())

    if not df_final.empty:
        df_final.to_csv("lupus_mean_expression.csv", index=False)
        print("Data saved to lupus_mean_expression.csv")
    else:
        print("No data to save - dataframe is empty")
        
    print(df_final)
    
    return f


def get_mean_expression_cm_cells(X):
    """
    Get final dataframe with mean gene expression for each sample for CM cells across all genes.
    
    Args:
        X: AnnData object with lupus data
    
    Returns:
        pd.DataFrame: Final dataframe with mean expression per sample for all genes
    """
    # Get expression data for all genes
    df = X.to_df()
    
    # Add metadata
    df["Status"] = X.obs["SLE_status"].values
    df["Condition"] = X.obs["Condition"].values
    df["Cell Type"] = X.obs["Cell Type"].values

    # Filter to only CM cells (try both "CM" and "cM" in case of different naming)
    df_filtered = df[df["Cell Type"].isin(["cM"])]
    print(f"After filtering for CM cells: {len(df_filtered)} rows")
    
    if len(df_filtered) == 0:
        print("No CM cells found! Available cell types:")
        print(df["Cell Type"].unique())
        return pd.DataFrame()
    
    # Get all gene columns (exclude metadata columns)
    metadata_cols = ["Status", "Condition", "Cell Type"]
    gene_cols = [col for col in df_filtered.columns if col not in metadata_cols]
    
    print(f"Processing {len(gene_cols)} genes")
    
    # Group by sample (Condition) and cell type to get mean expression per sample for each gene
    groupby_cols = ["Status", "Cell Type", "Condition"]
    df_final = df_filtered.groupby(groupby_cols, observed=False)[gene_cols].mean().reset_index()
    df_final = df_final.dropna().sort_values(["Cell Type", "Condition"])
    
    # Add cell count information
    df_count = df_filtered.groupby(["Cell Type", "Condition"], observed=False).size().reset_index(
        name="Cell Count").sort_values(["Cell Type", "Condition"])

    # Merge cell count with mean expression data
    df_final = df_final.merge(df_count, on=["Cell Type", "Condition"], how="left")
    df_final['Cell Type'] = df_final['Cell Type'].astype('category').cat.remove_unused_categories()
    
    return df_final
    



