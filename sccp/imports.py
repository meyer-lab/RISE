import anndata
import pandas as pd
from parafac2.normalize import prepare_dataset

from .gating import gateThomsonCells


def import_thomson() -> anndata.AnnData:
    """Import Thompson lab PBMC dataset."""
    # Cell barcodes, sample id of treatment and sample number
    metafile = pd.read_csv("sccp/data/Thomson/meta.csv", usecols=[0, 1])
    # X = sc.read_10x_mtx(
    #     "/opt/andrew/Thomson/", var_names="gene_symbols", make_unique=True
    # )
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

    return prepare_dataset(X, "Condition", geneThreshold=0.01)


def import_lupus(geneThreshold: float = 0.1) -> anndata.AnnData:
    """Import Lupus PBMC dataset.

    -- columns from observation data:
    {'batch_cov': POOL (1-23) cell was processed in,
    'ind_cov': patient cell was derived from,
    'Processing_Cohort': BATCH (1-4) cell was derived from,
    'louvain': louvain cluster group assignment,
    'cg_cov': broad cell type,
    'ct_cov': lymphocyte-specific cell type,
    'L3': marks a balanced subset of batch 4 used for model training,
    'ind_cov_batch_cov': combination of patient and pool, proxy for sample ID,
    'Age':	age of patient,
    'Sex': sex of patient,
    'pop_cov': ancestry of patient,
    'Status': SLE status: healthy, managed, treated, or flare,
    'SLE_status': SLE status: healthy or SLE}

    """
    X = anndata.read_h5ad("/opt/andrew/lupus/lupus.h5ad")
    X = anndata.AnnData(X.raw.X, X.obs, X.raw.var, X.uns, X.obsm)

    protein = anndata.read_h5ad("/opt/andrew/lupus/Lupus_study_protein_adjusted.h5ad")
    protein_df = protein.to_df()

    # Rename columns
    X.obs = X.obs.rename(
        {
            "batch_cov": "pool",
            "ind_cov": "patient",
            "cg_cov": "Cell Type",
            "ct_cov": "cell_type_lympho",
            "ind_cov_batch_cov": "Condition",
            "Age": "age",
            "Sex": "sex",
            "pop_cov": "ancestry",
        },
        axis=1,
    )

    X.obs = X.obs.merge(protein_df, how="left", left_index=True, right_index=True)

    # Get rid of IGTB1906_IGTB1906:dmx_count_AHCM2CDMXX_YE_0831 (Only 3 cells)
    X = X[X.obs["Condition"] != "IGTB1906_IGTB1906:dmx_count_AHCM2CDMXX_YE_0831"]

    return prepare_dataset(X, "Condition", geneThreshold=geneThreshold)
