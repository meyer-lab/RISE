import numpy as np
import pandas as pd
import scipy.io
import anndata
import os
from os.path import join
from .scRNA import tensorFy

path_here = os.path.dirname(os.path.dirname(__file__))


def combine_all_citeseq(saveAdata = False):
    """Imports 5 datasets from Hamad CITEseq """
    # Initiates import for control 
    features = pd.read_csv("/opt/andrew/HamadCITEseq/control/features.tsv.gz", sep="\t", header=None)
    data = pd.DataFrame(scipy.io.mmread("/opt/andrew/HamadCITEseq/control/matrix.mtx.gz").todense())

    # Keep information about type of expression
    data["Expression Type"] = features.iloc[:, 2].values
    data["Expression Name"] = features.iloc[:, 1].values
    
    # Keep only gene expression
    geneDF = data.loc[data["Expression Type"] == "Gene Expression"].drop(columns="Expression Type").reset_index(drop=True) 
    geneNames = geneDF["Expression Name"].values
    
    genesAll = np.transpose(geneDF.drop(columns="Expression Name")).to_numpy()
    numCells = [genesAll.shape[0]] # Save number of cells per experiment
    files = ["ic_pod1", "ic_pod7", "sc_pod1", "sc_pod1"]
    
    # Repeat process for all files and combine datasets
    for i in range(len(files)):
        features = pd.read_csv("/opt/andrew/HamadCITEseq/"+files[i]+"/features.tsv.gz", sep="\t", header=None)
        data = pd.DataFrame(scipy.io.mmread("/opt/andrew/HamadCITEseq/"+files[i]+"/matrix.mtx.gz").todense())
    
        data["Expression Type"] = features.iloc[:, 2].values
        
        geneDF = data.loc[data["Expression Type"] == "Gene Expression"].drop(columns="Expression Type").reset_index(drop=True)
        geneMatrix = np.transpose(geneDF).to_numpy()
        
        # Combines datasets and save number of cells per exp
        genesAll = np.vstack((genesAll,geneMatrix))
        numCells = np.append(numCells, geneMatrix.shape[0])
        
    # Save condition information in AnnData file
    files = ["control", "ic_pod1", "ic_pod7", "sc_pod1", "sc_pod7"]
    adata = anndata.AnnData(X = genesAll)
    adata.var_names = geneNames
    adata.obs["Condition"] = np.repeat(files, numCells)
    
    if saveAdata is False:
        return adata
    
    else:
        adata.write_h5ad(join(path_here, "data/HamadCITEseq.h5ad"))
    

def import_citeseq():
    """Normalizes 5 datasets from Hamad CITEseq and imports as tensory"""
    X = anndata.read_h5ad("/opt/andrew/HamadCITEseq/CITEseqCombined.h5ad")

    X.X = np.asarray(X.X, dtype=float)

    scalingfactor = 1000

    assert np.all(np.isfinite(X.X.data))

    X = X[:, np.mean(X.X > 0, axis=0) > 0.001]
    X.X /= np.sum(X.X, axis=0)
    X.X = np.log10((scalingfactor * X.X) + 1)

    # Center the genes
    X.X -= np.mean(X.X, axis=0)

    # Assign cells a count per-experiment so we can reindex
    return tensorFy(X, "Condition")
    