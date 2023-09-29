import numpy as np
import pandas as pd
import scipy.io
import anndata
import os
from os.path import join
from .scRNA import tensorFy
import scanpy as sc
import anndata

path_here = os.path.dirname(os.path.dirname(__file__))


def import_citeseq():
    """Imports 5 datasets from Hamad CITEseq"""
    files = ["control", "ic_pod1", "ic_pod7", "sc_pod1", "sc_pod7"]
    for i in range(len(files)):
        if i == 0:
            totalAnn = sc.read_10x_mtx("/opt/andrew/HamadCITEseq/"+files[i], gex_only=False)
            totalAnn.obs["Condition"] = np.repeat(files[i], np.shape(totalAnn)[0])   
    
        else:
            Ann = sc.read_10x_mtx("/opt/andrew/HamadCITEseq/"+files[i], gex_only=False)   
            Ann.obs["Condition"] = np.repeat(files[i], np.shape(Ann)[0])
            totalAnn = anndata.concat([totalAnn, Ann],merge="same")
    

    annGene = totalAnn[:, totalAnn.var["feature_types"] == "Gene Expression"]
    # A 32-bit float is high enough precision and uses 50% of the memory
    annGene.X.data = np.asarray(annGene.X.data, dtype=np.float32)
    
    # sc.pp.filter_cells(annGene, min_genes=200)
    sc.pp.filter_genes(annGene, min_cells=1)  
    sc.pp.normalize_total(annGene)
    sc.pp.log1p(annGene)
    sc.pp.highly_variable_genes(annGene, n_top_genes=10000)

    assert np.all(np.isfinite(annGene.X.data))

    # Center the genes
    annGene.X -= np.mean(annGene.X, axis=0)
    
    
    annProtein = totalAnn[:, totalAnn.var["feature_types"] == "Antibody Capture"]
    # A 32-bit float is high enough precision and uses 50% of the memory
    annProtein.X.data = np.asarray(annProtein.X.data, dtype=np.float32)
    annProtein.X.data -= np.nanmean(annProtein.X.data, axis=0)
    annProtein.X.data /= np.nanstd(annProtein.X.data, axis=0)
    protDF = annProtein.to_df().reset_index(drop=True)
    protDF["Condition"] = annProtein.obs["Condition"].values
    
    return tensorFy(annGene, "Condition"),  protDF

  
   
        
        
    
 