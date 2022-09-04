"""
# This creates Figure 12, typing Thompson drug data.
# """
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .common import subplotLabel, getSetup
from gmm.scImport import gene_import
import scanpy as sc
from copy import copy


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((15, 15), (3, 3))
    subplotLabel(ax)

    # Add subplot labels
    geneDF = gene_import(offset=1, filter=False)
    drugCol = geneDF.Drug
    geneDF.drop(columns=["Drug"], axis=1, inplace=True)
    adata = sc.AnnData(geneDF)
    sc.pp.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=0.75)
    sc.tl.rank_genes_groups(adata, groupby='leiden')
    marker_matches = sc.tl.marker_gene_overlap(adata, marker_genes)
    print(marker_matches)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=['leiden'], ax=ax[0])
    savedata = copy(adata)
    savedata.obs = savedata.obs.replace(clust_names)
    savedata.obs.columns = ["Cell Type"]
    savedata.obs.to_csv("gmm/data/ThompsonCellTypes.csv")
    adata.rename_categories('leiden', clust_list)
    adata.obs["Drug"] = drugCol.values
    sc.pp.subsample(adata, fraction=0.1, random_state=0)
    sc.pl.umap(adata, color='leiden', legend_loc='on data', title='', frameon=False, ax=ax[1])
    sc.pl.umap(adata, color='Drug', title='', legend_loc='on data', legend_fontsize=3, frameon=False, ax=ax[2])
    sc.pl.umap(adata, color=['CD14'], ax=ax[3])
    sc.pl.umap(adata, color=['LAD1'], ax=ax[4])
    sc.pl.umap(adata, color=['MS4A1'], ax=ax[5])
    sc.pl.umap(adata, color=['CD3D'], ax=ax[6])
    sc.pl.umap(adata, color=['IL2RA'], ax=ax[7])
    sc.pl.umap(adata, color=['NKG7'], ax=ax[8])

    return f


clust_names = {
    "0": "NK",
    "1": "Monocytes",
    "2": "Monocytes",
    "3": "Monocytes",
    "4": "None",
    "5": "T Cells",
    "6": "CD8 T Cells",
    "7": "NK",
    "8": "T Cells ",
    "9": "T helpers",
    "10": "Dendritic Cells",
    "11": "B Cells",
    "12": "Monocytes",
    "13": "Monocytes",
    "14": "Monocytes"}
clust_list = [
    "NK",
    "Monocytes ",
    "Monocytes  ",
    "Monocytes   ",
    "None",
    "T Cells",
    "CD8 T Cells",
    "NK ",
    "T Cells ",
    "T helpers",
    "Dendritic Cells",
    "B Cells",
    " Monocytes",
    "  Monocytes",
    "   Monocytes"]


marker_genes = {
    'Monocytes': [
        'CD14',
        'CD33',
        'LYZ',
        'LGALS3',
        'CSF1R',
        'ITGAX',
        'HLA-DRB1'],
    'Dendritic cells': [
        'LAD1',
        'LAMP3',
        'TSPAN13',
        'CLIC2',
        'FLT3'],
    'B-cells': [
        'MS4A1',
        'CD19',
        'CD79A'],
    'T-helpers': [
        'TNF',
        'TNFRSF18',
        'IFNG',
        'IL2RA',
        'BATF'],
    'T cells': [
        'CD27',
        'CD69',
        'CD2',
        'CD3D',
        'CXCR3',
        'CCL5',
        'IL7R',
        'CXCL8',
        'GZMK'],
    'Natural Killers': [
        'NKG7',
        'GNLY',
        'PRF1',
        'FCGR3A',
        'NCAM1',
        'TYROBP'],
    'CD8': [
        'CD8A',
        'CD8B']
}
