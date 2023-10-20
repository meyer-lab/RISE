import numpy as np
import anndata
import scanpy as sc


def import_citeseq() -> anndata.AnnData:
    """Imports 5 datasets from Hamad CITEseq."""
    # files = ["control", "ic_pod1", "ic_pod7", "sc_pod1", "sc_pod7"]

    # data = {
    #     k: sc.read_10x_mtx("/opt/andrew/HamadCITEseq/" + k, gex_only=False, make_unique=True) for k in files
    # }
    # X = anndata.concat(data, merge="same", label="Condition")

    # sc.pp.filter_genes(X, min_cells=100)
    # sc.pp.normalize_total(X)
    # sc.pp.log1p(X)
    # sc.pp.highly_variable_genes(X, n_top_genes=4000)

    # X = X[:, X.var["highly_variable"]]

    # X.write_h5ad("HamadCITEseq4000genes.h5ad", compression="gzip")

    X = anndata.read_h5ad("/opt/andrew/HamadCITEseq4000genes.h5ad")

    X.X = X.X.todense()

    # Center and read normalize the genes
    X.X /= np.sum(X.X, axis=0)
    X.X -= np.mean(X.X, axis=0)

    return X
