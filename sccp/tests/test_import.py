"""
Test the cross validation accuracy.
"""
import pytest
import numpy as np
from ..imports import import_thomson, import_lupus, import_citeseq
import pandas as pd
from ..gating import gateThomsonCells
import anndata


@pytest.mark.parametrize("import_func", [import_thomson, import_lupus, import_citeseq])
def test_imports(import_func):
    """Test import functions."""
    X = import_func()
    print(f"Data shape: {X.shape}")
    assert X.X.dtype == np.float32


def test_GateThomson():
    """Test that gating function matches size of data"""
    metafile = pd.read_csv("sccp/data/Thomson/meta.csv", usecols=[0, 1])
    X = anndata.read_h5ad("/opt/andrew/thomson_raw.h5ad")
    obs = X.obs.reset_index(names="cell_barcode")

    metafile = pd.merge(
        obs, metafile, on="cell_barcode", how="left", validate="one_to_one"
    )

    X.obs = pd.DataFrame(
        {
            "cell_barcode": metafile["cell_barcode"],
            "Condition": pd.Categorical(metafile["sample_id"]),
        }
    )

    doubletDF = pd.read_csv("sccp/data/Thomson/ThomsonDoublets.csv", index_col=0)
    doubletDF.index.name = "cell_barcode"
    X.obs = X.obs.join(doubletDF, on="cell_barcode", how="inner")

    singlet_indices = X.obs.loc[X.obs["doublet"] == 0].index.values
    X.obs = X.obs.reset_index(drop=True)
    X = X[singlet_indices, :]
    
    X.obs = X.obs.set_index("cell_barcode")
    gateThomsonCells(X)
