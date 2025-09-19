"""
Figure 5a_e Generation Script

This module generates a comprehensive figure comparing PCA and PF2 component analyses 
for gene loadings and factors.
"""

import anndata
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import mygene
from .common import getSetup, subplotLabel
import statsmodels.api as sm


def makeFigure():
    ax, f = getSetup((22, 9), (4, 10))
    subplotLabel(ax)
    
    geneAmount = 10
    X = anndata.read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    
    pc1_load = load_pc_loadings(pc_component=1)
    pc2_load = load_pc_loadings(pc_component=2)
    
    # Get the final dataframe with mean gene expression for each sample 
    # for top/bottom weighted overlapping genes in nCM and CM populations only
    df_final, selected_genes = get_mean_expression_overlapping_genes(X, pc1_load, pc2_load, geneAmount)

    df_final.to_csv("lupus_mean_expression.csv", index=False)

    print(df_final)
    print(f"Final dataframe shape: {df_final.shape}")
    print(f"Number of selected genes: {len(selected_genes)}")
    print(f"Selected genes: {selected_genes}")
    print("\nFinal dataframe with mean gene expression per sample:")
    print(df_final.head(10))
    print(f"\nColumns in final dataframe: {list(df_final.columns)}")
    
    return f



def load_pc_loadings(pc_component: int) -> pd.DataFrame:
    """
    Load and preprocess PC loadings for a specific component.

    Args:
        pc_component (int): Principal Component number (1 or 2)
        geneAmount (int): Number of genes to consider

    Returns:
        pd.DataFrame: Processed PC loadings DataFrame
    """

    df = pd.read_csv(f"loadings_time_series_PC{pc_component}.csv", dtype=str)
    df = df.rename(columns={"Unnamed: 0": "Gene"})
    df["Gene"] = convert_gene_symbols(df["Gene"])[0]
    df[f"PC{pc_component}"] = stats.zscore(df[f"PC{pc_component}"].astype(float))

    return df

def load_pf2_loadings(X) -> pd.DataFrame:
    """
    Load and preprocess PC loadings for a specific component.

    Args:
        pc_component (int): Principal Component number (1 or 2)
        geneAmount (int): Number of genes to consider

    Returns:
        pd.DataFrame: Processed PC loadings DataFrame
    """

    df = pd.DataFrame(data=X.varm['Pf2_C'], columns=[f"Pf2_{i}" for i in range(1, X.varm['Pf2_C'].shape[1]+1)])
    df =  df.set_index(X.var_names).reset_index().rename(columns={"index": "Gene"})

    return df
    


def convert_gene_symbols(genes):
    """
    Convert gene symbols using MyGene.

    Args:
        genes (list): List of gene symbols
        species (str): Species for gene conversion

    Returns:
        tuple: Converted genes and list of genes without hits
    """
    mg = mygene.MyGeneInfo()
    results = mg.querymany(
        genes, 
        scopes='symbol', 
        fields=['symbol', 'entrezgene'], 
        species='mouse', 
        transformed=True
    )

    conversion_map = []
    no_hit_genes = []
    
    for gene, result in zip(genes, results):
        if result and 'symbol' in result:
            conversion_map.append(result.get('symbol', gene).upper())
        else:
            conversion_map.append(gene.upper())
            no_hit_genes.append(gene)
    
    return conversion_map, no_hit_genes


def get_mean_expression_overlapping_genes(X, pc1_load, pc2_load, geneAmount=10):
    """
    Get final dataframe with mean gene expression for each sample for top/bottom weighted overlapping genes
    in nCM and CM populations only.
    
    Args:
        X: AnnData object with lupus data
        pc1_load: PC1 loadings DataFrame
        pc2_load: PC2 loadings DataFrame
        geneAmount: Number of top and bottom genes to include
    
    Returns:
        pd.DataFrame: Final dataframe with mean expression per sample for selected genes
        list: List of selected gene names
    """
    # Find overlapping genes between PC loadings and dataset
    pc1_genes = set(pc1_load['Gene'])
    pc2_genes = set(pc2_load['Gene'])
    dataset_genes = set(X.var_names)
    
    # Get genes that overlap between PC1, PC2, and the dataset
    overlap_genes = list(pc1_genes & pc2_genes & dataset_genes)
    
    print(f"Found {len(overlap_genes)} overlapping genes")
    
    if not overlap_genes:
        raise ValueError("No overlapping genes found between PC loadings and dataset")
    
    # Get top and bottom weighted genes from PC1 and PC2
    selected_genes = []
    
    for pc_load, pc_name in [(pc1_load, "PC1"), (pc2_load, "PC2")]:
        # Filter to overlapping genes only
        pc_overlap = pc_load[pc_load['Gene'].isin(overlap_genes)].copy()
        
        # Sort by PC weights and get top and bottom genes
        pc_sorted = pc_overlap.sort_values(by=pc_name, ascending=False)
        top_genes = pc_sorted['Gene'].head(geneAmount).tolist()
        bottom_genes = pc_sorted['Gene'].tail(geneAmount).tolist()
        
        selected_genes.extend(top_genes)
        selected_genes.extend(bottom_genes)
    
    # Remove duplicates while preserving order
    selected_genes = list(dict.fromkeys(selected_genes))
    
    print(f"Selected {len(selected_genes)} genes after filtering for top/bottom weighted")
    
    # Get expression data for selected genes
    genesV = X[:, selected_genes]
    df = genesV.to_df()
    
    # Add metadata
    df["Status"] = X.obs["SLE_status"].values
    df["Condition"] = X.obs["Condition"].values
    df["Cell Type"] = X.obs["Cell Type"].values
    
    # Filter to only nCM and CM populations
    df = df[df["Cell Type"].isin(["nCM", "CM"])]
    
    # Group by sample (Condition) and cell type to get mean expression per sample for each gene
    groupby_cols = ["Status", "Cell Type", "Condition"]
    gene_cols = selected_genes
    df_final = df.groupby(groupby_cols, observed=False)[gene_cols].mean().reset_index()
    df_final = df_final.dropna().sort_values(["Cell Type", "Condition"])
    
    # Add cell count information
    df_count = df.groupby(["Cell Type", "Condition"], observed=False).size().reset_index(
        name="Cell Count").sort_values(["Cell Type", "Condition"])
    df_count = df_count[df_count["Cell Type"].isin(["nCM", "CM"])]

    # Merge cell count with mean expression data
    df_final = df_final.merge(df_count, on=["Cell Type", "Condition"], how="left")
    df_final['Cell Type'] = df_final['Cell Type'].astype('category').cat.remove_unused_categories()
    
    return df_final, selected_genes


def avegene_lupus(X, genes, overlap_genes):
    """Average gene expression of multiple overlapping genes in lupus samples for nCM and CM populations."""
    # Handle both single gene (string) and multiple genes (list)
    if isinstance(genes, str):
        genes = [genes]
    
    # Filter genes to only include those that overlap and are present in the dataset
    available_genes = [gene for gene in genes if gene in overlap_genes and gene in X.var_names]
    
    if not available_genes:
        raise ValueError(f"None of the provided genes {genes} are found in the overlap genes or dataset")
    
    # Get expression data for the available overlapping genes
    genesV = X[:, available_genes]
    df = genesV.to_df()
    
    # Add metadata
    df["Status"] = X.obs["SLE_status"].values
    df["Condition"] = X.obs["Condition"].values
    df["Cell Type"] = X.obs["Cell Type"].values
    
    # Filter to only nCM and CM populations
    df = df[df["Cell Type"].isin(["nCM", "CM"])]
    
    # If multiple genes, compute their average expression
    if len(available_genes) > 1:
        # Create a new column with the average expression across overlapping genes
        gene_avg_name = f"avg_{len(available_genes)}_overlapping_genes"
        df[gene_avg_name] = df[available_genes].mean(axis=1)
        # Keep only the average column and metadata
        expression_cols = [gene_avg_name]
    else:
        # Single gene case
        gene_avg_name = available_genes[0]
        expression_cols = available_genes
    
    # Group by sample (Condition) and cell type to get mean expression per sample
    groupby_cols = ["Status", "Cell Type", "Condition"]
    df_mean = df.groupby(groupby_cols, observed=False)[expression_cols + available_genes].mean().reset_index()
    df_mean = df_mean.dropna().sort_values(["Cell Type", "Condition"])
    
    # Add cell count information
    df_count = df.groupby(["Cell Type", "Condition"], observed=False).size().reset_index(
        name="Cell Count").sort_values(["Cell Type", "Condition"])
    df_count = df_count[df_count["Cell Type"].isin(["nCM", "CM"])]

    # Merge cell count with mean expression data
    df_final = df_mean.merge(df_count, on=["Cell Type", "Condition"], how="left")
    df_final['Cell Type'] = df_final['Cell Type'].astype('category').cat.remove_unused_categories()
    
    return df_final, gene_avg_name, available_genes


