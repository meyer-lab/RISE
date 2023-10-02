import numpy as np
import anndata
import scanpy as sc
from .scRNA import tensorFy


def import_citeseq():
    """Imports 5 datasets from Hamad CITEseq"""
    files = ["control", "ic_pod1", "ic_pod7", "sc_pod1", "sc_pod7"]

    totalAnn = [
        sc.read_10x_mtx("/opt/andrew/HamadCITEseq/" + f, gex_only=False) for f in files
    ]
    for i in range(len(totalAnn)):
        totalAnn[i].obs["Condition"] = np.repeat(files[i], totalAnn[i].shape[0])

    totalAnn = anndata.concat(totalAnn, merge="same")

    annGene = totalAnn[:, totalAnn.var["feature_types"] == "Gene Expression"]

    # A 32-bit float is high enough precision and uses 50% of the memory
    annGene.X.data = np.asarray(annGene.X.data, dtype=np.float32)

    # sc.pp.filter_cells(annGene, min_genes=200)
    sc.pp.filter_genes(annGene, min_cells=10)
    sc.pp.normalize_total(annGene)
    sc.pp.log1p(annGene)
    sc.pp.highly_variable_genes(annGene, n_top_genes=10000)

    assert np.all(np.isfinite(annGene.X.data))

    # Center the genes
    annGene.X -= np.mean(annGene.X, axis=0)

    print(annGene.shape)

    annProtein = totalAnn[:, totalAnn.var["feature_types"] == "Antibody Capture"]
    annProtein.X.data -= np.nanmean(annProtein.X.data, axis=0)
    annProtein.X.data /= np.nanstd(annProtein.X.data, axis=0)
    protDF = annProtein.to_df().reset_index(drop=True)
    protDF["Condition"] = annProtein.obs["Condition"].values

    return tensorFy(annGene, "Condition"), protDF
