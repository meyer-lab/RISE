import numpy.typing as npt
import scanpy
import pandas as pd


def gateThomsonCellsLeiden(X) -> npt.ArrayLike:
    """Manually gates cell types for Thomson UMAP"""
    scanpy.pp.neighbors(X, n_neighbors=15, use_rep="projections", random_state=0)
    scanpy.tl.leiden(X, resolution=3, random_state=0)
    X.obs["Cell Type"] = X.obs.leiden.replace(thomson_layer1).astype(str)
    X.obs["Cell Type2"] = X.obs.leiden.replace(thomson_layer2).astype(str)

    return X


def gateThomsonCells(X) -> npt.ArrayLike:
    """Manually gates cell types for Thomson UMAP"""
    cellTypeDF = pd.read_csv("sccp/data/Thomson/ThomsonCellTypes.csv", index_col=0)
    X.obs["Cell Type"] = cellTypeDF["Cell Type"].values.astype(str)
    X.obs["Cell Type2"] = cellTypeDF["Cell Type2"].values.astype(str)

    return X


thomson_layer1 = {
    "0": "NK Cells",
    "1": "T Cells",
    "2": "Monocytes",
    "3": "Monocytes",
    "4": "Monocytes",
    "5": "Monocytes",
    "6": "Monocytes",
    "7": "Monocytes",
    "8": "Monocytes",
    "9": "Monocytes",
    "10": "Monocytes",
    "11": "Monocytes",
    "12": "Monocytes",
    "13": "Monocytes",
    "14": "Monocytes",
    "15": "Monocytes",
    "16": "Monocytes",
    "17": "Monocytes",
    "18": "Monocytes",
    "19": "Monocytes",
    "20": "Monocytes",
    "21": "Monocytes",
    "22": "Monocytes",
    "23": "Monocytes",
    "24": "Monocytes",
    "25": "DCs",
    "26": "Monocytes",
    "27": "B Cells",
    "28": "Monocytes",
    "29": "Monocytes",
    "30": "Monocytes",
    "31": "Monocytes",
    "32": "Monocytes",
    "33": "T Helpers",
    "34": "Monocytes",
    "35": "DCs",
    "36": "Monocytes",
    "37": "Monocytes",
    "38": "Monocytes",
    "39": "T Cells",
    "40": "Monocytes",
    "41": "Monocytes",
    "42": "Monocytes",
    "43": "B Cells",
    "44": "Monocytes",
}


thomson_layer2 = {
    "0": "NK Cells",
    "1": "Memory T Cells",
    "2": "Monocytes",
    "3": "Myeloid Suppressors",
    "4": "Macrophages",
    "5": "Monocytes",
    "6": "Monocytes",
    "7": "Monocytes",
    "8": "Macrophages",
    "9": "Monocytes",
    "10": "Monocytes",
    "11": "Monocytes",
    "12": "Macrophages",
    "13": "Monocytes",
    "14": "Monocytes",
    "15": "Macrophages",
    "16": "Monocytes",
    "17": "Monocytes",
    "18": "Monocytes",
    "19": "Monocytes",
    "20": "Monocytes",
    "21": "Macrophages",
    "22": "Macrophages",
    "23": "Monocytes",
    "24": "Macrophages",
    "25": "pDCs",
    "26": "Macrophages",
    "27": "Memory B",
    "28": "Fibroblasts",
    "29": "Activated Monocyte",
    "30": "Monocytes",
    "31": "Fibroblasts",
    "32": "Monocytes",
    "33": "Cytotoxic T",
    "34": "Myeloid Suppressors",
    "35": "cDCs",
    "36": "Monocytes",
    "37": "Monocytes",
    "38": "Monocytes",
    "39": "CD4 T Cells",
    "40": "Activated Monocyte",
    "41": "None",
    "42": "Macrophages",
    "43": "Naive B",
    "44": "Monocytes",
}


# Taken directly from PopAlign and used to annotate level 1
marker_genes_1 = {
    "Monocytes": [
        "CD14",
        "CD33",
        "LYZ",
        "FCER1G",
        "LGALS3",
        "CSF1R",
        "ITGAX",
        "ITGAM",
        "CD86",
        "HLA-DRB1",
    ],
    "Dendritic cells": ["LAD1", "LAMP3", "TSPAN13", "CLIC2", "FLT3"],
    "B-cells": ["MS4A1", "CD19", "CD79A"],
    "T-helpers": ["TNF", "TNFRSF18", "IFNG", "IL2RA", "BATF"],
    "T cells": [
        "CD27",
        "CD69",
        "CD2",
        "CD3D",
        "CXCR3",
        "CCL5",
        "IL7R",
        "CXCL8",
        "GZMK",
    ],
    "Natural Killers": ["NKG7", "GNLY", "PRF1", "FCGR3A", "NCAM1", "TYROBP"],
}


# This is a list of genes which I assembled to label the second layer of cell annotation. I assembled this using several sources.
marker_genes_2 = {
    "B cells": ["PXK", "MS4A1", "CD19", "CD74", "CD79A", "BANK1", "PTPRC", "CR2"],
    "B Memory": ["NPIB15", "BACH2", "IL7", "NMBR", "MS4A1", "MBL2", "LY86" "CD27"],
    "B Naive": ["P2RX5", "SIK1", "SLC12A1", "SELL", "RALGPS2", "PTPRCAP", "PSG2"],
    "Basophils": ["CCL4", "NPL", "WRN", "NFIL3", "TEC", "OTUB2", "FAR2"],
    "DCs": ["ITGAX", "ZBTB46", "LAMP3", "CXCR1", "ITGAM", "FCER1A", "IL6"],
    "Macrophages": ["CD68", "FCGR1", "NAAA", "JAML", "TYROBP", "LYZ2", "H2-DMA"],
    "Monocytes": [
        "APOBEC3A",
        "LYZ",
        "CD14",
        "CFP",
        "HLA-DRA",
        "S100A9",
        "S100A8",
        "CSF3R",
    ],
    "Myeloid Suppressors": [
        "S100A4",
        "S100A9",
        "ICAM1",
        "S100A8",
        "ITGAM",
        "LY6G",
        "GR1",
        "FCGR3A",
    ],
    "NK": ["NKG7", "GNLY", "KLRD1", "KLRF1", "NCR1", "DOCK2", "GZMA", "IRF7"],
    "pDCs": ["BST2", "CLEC4C", "MAP3K2", "KLK1", "CMAH", "TRADD", "LILRA4"],
    "T Cells": ["TRBC2", "CD3D", "CD3G", "CD3E", "LTB", "IL7R", "LEF1"],
    "Cytotoxic T": ["TRAC", "CD8A", "GZMB", "CD2", "CD27", "CD5", "CD27"],
    "Helper T": ["CCR4", "CD4", "IL13", "CD28", "CD3G", "IL2", "CCR6"],
    "Memory T": ["CCR7", "CD2", "PTPRC", "CD28", "LEF1", "S100A8", "GIMAP4"],
}
