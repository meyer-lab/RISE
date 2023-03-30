from os.path import join, dirname
import numpy as np
import pandas as pd
import csv
import xarray as xa
from scipy.io import mmread
from scipy.stats import linregress
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn import preprocessing
import anndata


path_here = dirname(dirname(__file__))


def import_perturb_RPE(limit_cells=100):
    ds_disk = anndata.read_h5ad("/opt/andrew/rpe1_normalized_singlecell_01.h5ad")

    sgRNAs = ds_disk.obs_vector("sgID_AB")
    sgUnique, sgIndex, sgCounts = np.unique(
        sgRNAs, return_inverse=True, return_counts=True
    )

    X = xa.DataArray(
        data=np.zeros((len(sgUnique), limit_cells, ds_disk.shape[1]), dtype=np.float32),
        dims=["sgRNA", "Cells", "Genes"],
        coords=[
            sgUnique,
            np.arange(limit_cells),
            ds_disk.var_vector("gene_name"),
        ],
    )

    for sgi in range(len(sgUnique)):
        x_temp = ds_disk[sgIndex == sgi, :]
        assert x_temp.shape[0] == sgCounts[sgi]

        if x_temp.shape[0] > limit_cells:
            x_temp = x_temp[:limit_cells, :]

        X[sgi, 0 : x_temp.shape[0], :] = x_temp.X.toarray()

    # These genes have nans for some reason
    X = xa.concat([X[:, :, 0:773], X[:, :, 774:]], dim="Genes")
    X = xa.concat([X[:, :, 0:7002], X[:, :, 7003:]], dim="Genes")

    # Do not allow problematic values
    assert np.all(np.isfinite(X))
    return X


def import_thompson_drug():
    """Imports cell readings from scRNA-seq data from PBMCs from PopAlign paper"
    -Description of each file-
    drugScreeen : str Path to a sparse matrix
    barcodes : str Path to a .tsv 10X barcodes file
    metafile : str Path to a metadata file. Must contains `cell_barcodes` and `sample_id` columns
    genes : str Path to a .tsv 10X gene file"""

    # Cell barcodes, sample id of treatment and sample number (33482, 3)
    metafile = pd.read_csv("sccp/data/meta.csv")
    # Cell barcodes (33482)
    barcodes = pd.read_csv("sccp/data/barcodes.tsv", sep="\t", header=None, names=("cell_barcode", )).reset_index()

    genes = pd.read_csv("sccp/data/features.tsv", sep="\t", names=("ENSG", "Name", "Null"))  # Gene Names (32738)

    metafile = pd.merge(metafile, barcodes, on="cell_barcode", how='left', validate="one_to_one")

    # Sparse matrix of each cell/genes (32738,33482)-(Genes,Cell)
    drugScreen = mmread("/opt/andrew/drugscreen.mtx").toarray()  
    drugScreen = drugScreen.astype(np.float64)

    df = pd.DataFrame(drugScreen.T, index=np.arange(drugScreen.shape[1]), columns=genes["Name"]).reset_index()

    df_full = pd.merge(df, metafile, on="index", how='left', validate="one_to_one")
    df_full = df_full.rename(columns={"sample_id": "Drug"})

    return df_full.drop(columns=["cell_barcode", "sample_number", "index"])


def mu_sigma_normalize(df, scalingfactor=1000):
    """Calculates the mu and sigma for every gene and returns
    means, sigmas, and dataframe filtered for genes expressed
    in > 0.1% of cells."""
    X = df.drop("Drug", axis=1).to_numpy()
    assert np.all(np.isfinite(X))

    keepGenes = np.mean(X > 0, axis=0) > 0.001
    normG = X / np.sum(X, axis=0, keepdims=True)

    logG = np.log10((scalingfactor * normG) + 1)
    means = np.mean(logG, axis=0)
    cv = np.std(logG, axis=0) / means

    normDF = logG.iloc[:, np.append(keepGenes, True)]
    means = means[keepGenes]
    cv = cv[keepGenes]

    return normDF, np.log10(means + 1e-10), np.log10(cv + 1e-10)


def gene_filter(geneDF, mean, std, offset_value=1.0):
    """Filters genes whos variance are higher than woudl be predicted by a Poisson distribution"""
    slope, intercept, _, _, _ = linregress(mean, std)
    inter = intercept + np.log10(offset_value)

    above_idx = np.where(std > mean * slope + inter)
    finalDF = geneDF.iloc[
        :, np.append(np.asarray(above_idx).flatten(), geneDF.shape[1] - 1)
    ]

    return finalDF, above_idx


def gene_import(offset=1.0):
    """Imports gene data from PopAlign and perfroms gene filtering process"""
    genesDF = import_thompson_drug()
    filteredGeneDF, logmean, logstd = mu_sigma_normalize(genesDF, scalingfactor=1000)
    if offset != 1.0:
        filteredGeneDF, _ = gene_filter(
            filteredGeneDF, logmean, logstd, offset_value=offset
        )
    return filteredGeneDF


def ThompsonXA_SCGenes(saveXA=False, offset=1.0):
    """Turns filtered and normalized cells into an Xarray."""
    if saveXA is True:
        if offset == 1.0:
            df = pd.read_csv("/opt/andrew/scRNA_drugDF_NoOffset.csv")
            df = df.drop(columns=["Unnamed: 0"], axis=1)
        else:
            df = gene_import(offset=offset)

        df = df.sort_values(by=["Drug"])
        df = assign_celltype(df)

        # Assign cells a count per-experiment so we can reindex
        cellCount = df.groupby(by=["Drug"]).size().values
        df["Cell"] = np.concatenate([np.arange(int(cnt)) for cnt in cellCount])

        XA = df.set_index(["Cell", "Drug"]).to_xarray()
        celltypeXA = XA["Cell Type"]
        XA = XA.drop_vars(["Cell Type"])
        XA = XA.to_array(dim="Gene")

        XA = XA.transpose()
        celltypeXA = celltypeXA.transpose()

        XA.to_netcdf(join(path_here, "data/scRNA_drugXA.nc"))
        celltypeXA.to_netcdf(join(path_here, "data/scRNA_celltypeXA.nc"))

    else:
        if offset == 1.0:
            XA = xa.open_dataarray("/opt/andrew/scRNA_drugXA_NoOffset.nc")
            celltypeXA = xa.open_dataarray("/opt/andrew/scRNA_celltypeXA_NoOffset.nc")
        else:
            XA = xa.open_dataarray("/opt/andrew/scRNA_drugXA.nc")
    
    ### I *believe* that padding with zeros does not affect PARAFAC2 results.
    ### We should check this though.
    XA.values -= np.nanmean(XA.values, axis=(0,1), keepdims=True)       
    XA.values = np.nan_to_num(XA.values)       
    XA.name = "data"
    celltypeXA.name = "Cell Type"
    return xa.merge([XA, celltypeXA], compat="no_conflicts")


def assign_celltype(df):
    """Assignign cell types via scanpy and SVM"""
    import scanpy as sc

    celltypeDF = df.drop(columns=["Drug"], axis=1)
    genes_list = celltypeDF.columns
    adata = sc.AnnData(celltypeDF)
    sc.pp.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=0.75)
    sc.tl.rank_genes_groups(adata, groupby="leiden")
    marker_matches = sc.tl.marker_gene_overlap(adata, marker_genes)
    sc.tl.umap(adata)
    adata.obs = adata.obs.replace(clust_names)
    adata.obs.columns = ["Cell Type"]
    adata = drug_SVM(adata, genes_list)

    df["Cell Type"] = adata.obs.values

    return df.reset_index(drop=True)


def drug_SVM(save_data, genes):
    """Retrieves cell types from perturbed data"""
    completeDF = pd.DataFrame(data=save_data.X)
    completeDF.columns = genes
    completeDF["Cell Type"] = save_data.obs.values
    drugDF = completeDF.loc[
        (completeDF["Cell Type"] == "T Helpers") | (completeDF["Cell Type"] == "None")
    ]
    trainingDF = pd.DataFrame()
    for key in training_markers:
        cell_weight = (
            completeDF.loc[completeDF["Cell Type"] == key].shape[0]
            / completeDF.shape[0]
        )
        concatDF = drugDF.sort_values(by=training_markers[key]).tail(
            n=int(np.ceil(50 * cell_weight) + 5)
        )
        concatDF["Cell Type"] = key
        trainingDF = pd.concat([trainingDF, concatDF])

    le = preprocessing.LabelEncoder()
    le.fit(trainingDF["Cell Type"].values)
    svm = make_pipeline(preprocessing.StandardScaler(), SVC(gamma="auto"))
    svm.fit(
        trainingDF.drop("Cell Type", axis=1),
        le.transform(trainingDF["Cell Type"].values),
    )
    drugDF = drugDF.drop("Cell Type", axis=1)
    drugDF["Cell Type"] = le.inverse_transform(svm.predict(drugDF.values))
    save_data.obs.loc[
        (save_data.obs["Cell Type"] == "T Helpers")
        | (save_data.obs["Cell Type"] == "None"),
        "Cell Type",
    ] = drugDF["Cell Type"].values
    return save_data


clust_names = {
    "0": "NK",
    "1": "Monocytes",
    "2": "Monocytes",
    "3": "Monocytes",
    "4": "None",
    "5": "T Cells",
    "6": "CD8 T Cells",
    "7": "NK",
    "8": "T Cells",
    "9": "T Helpers",
    "10": "Dendritic Cells",
    "11": "B Cells",
    "12": "Monocytes",
    "13": "Monocytes",
    "14": "Monocytes",
}


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
    "   Monocytes",
]


marker_genes = {
    "Monocytes": ["CD14", "CD33", "LYZ", "LGALS3", "CSF1R", "ITGAX", "HLA-DRB1"],
    "Dendritic Cells": ["LAD1", "LAMP3", "TSPAN13", "CLIC2", "FLT3"],
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
    "CD8": ["CD8A", "CD8B"],
}

training_markers = {
    "Monocytes": ["CD14"],
    "Dendritic Cells": ["LAD1"],
    "B Cells": ["MS4A1"],
    "T Cells": ["CD3D"],
    "NK": ["NKG7"],
    "CD8 T Cells": ["CD8A"],
}
