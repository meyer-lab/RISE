"""Metrics used for Pf2 projections"""
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler

from .figures.common import flattenData, flattenProjs

DATA_DIR = Path(__file__).parent / "data"


METRICS = (
    "PCR batch",
    "Batch ASW",
    "graph connectivity",
    "graph iLISI",
    "kBET",
    "NMI cluster/label",
    "ARI cluster/label",
    "Cell type ASW",
    "graph cLISI",
    "isolated label F1",
    "isolated label silhouette",
)


def preprocess_pf2_pca_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Dataset"] = [method.split("/")[1] for method in df["Method"]]
    df["Output"] = [method.split("/")[-1].split("_")[1] for method in df["Method"]]
    df["Features"] = [
        {"full_feature": "FULL", "hvg": "HVG"}[method.split("/")[4]]
        for method in df["Method"]
    ]
    df["Scaling"] = [method.split("/")[3] for method in df["Method"]]
    df = df[~df["Method"].str.contains("unintegrated")]
    df["Rank"] = df["Rank"].astype(int)
    df["Method"] = [
        method.split("/")[-1].split("_")[0].upper() for method in df["Method"]
    ]
    dataset_translation = {
        "large_atac_gene_activity": "mouse_brain_atac_genes_large",
        "small_atac_gene_activity": "mouse_brain_atac_genes_small",
        "small_atac_peaks": "mouse_brain_atac_peaks_small",
    }
    df["Dataset"] = df["Dataset"].replace(dataset_translation)
    df["Method_Full"] = [
        f"{method}-{rank}_{output}_{scaling}_{features}"
        for method, rank, output, scaling, features in zip(
            df["Method"], df["Rank"], df["Output"], df["Scaling"], df["Features"]
        )
    ]
    df = df[df["Dataset"] != "pbmc3k_processed"]
    return df


def assemble_df():
    pf2_metrics = preprocess_pf2_pca_metrics(
        pd.read_excel(DATA_DIR / "Pf2-metrics.xlsx")
    )
    pca_metrics = preprocess_pf2_pca_metrics(
        pd.read_excel(DATA_DIR / "pca-metrics.xlsx").rename(
            columns={"Unnamed: 0": "Method"}
        )
    )
    scib_metrics_separate = pd.read_excel(
        DATA_DIR / "ScibMetrics.xlsx", sheet_name=None
    )
    scib_metrics = pd.concat(
        [
            scib_metrics_separate[ds]
            for ds in pf2_metrics["Dataset"].value_counts().index.values
        ]
    )
    scib_metrics["Method_Full"] = [
        f"{method}_{output}_{scaling}_{features}"
        for method, output, scaling, features in zip(
            scib_metrics["Method"],
            scib_metrics["Output"],
            scib_metrics["Scaling"],
            scib_metrics["Features"],
        )
    ]
    scib_metrics.drop(columns=["HVG conservation", "CC conservation"], inplace=True)
    scib_metrics = scib_metrics[~scib_metrics["Method"].str.contains("Pf2")]
    scib_metrics["Rank"] = -1
    metric_translation = {
        "NMI_cluster/label": "NMI cluster/label",
        "ARI_cluster/label": "ARI cluster/label",
        "ASW_label": "Cell type ASW",
        "ASW_label/batch": "Batch ASW",
        "PCR_batch": "PCR batch",
        "cell_cycle_conservation": "CC conservation",
        "isolated_label_F1": "isolated label F1",
        "isolated_label_silhouette": "isolated label silhouette",
        "graph_conn": "graph connectivity",
        "iLISI": "graph iLISI",
        "cLISI": "graph cLISI",
        "hvg_overlap": "HVG conservation",
        "trajectory": "trajectory conservation",
    }
    pf2_metrics.rename(columns=metric_translation, inplace=True)
    pca_metrics.rename(columns=metric_translation, inplace=True)
    common_columns = list(
        set(scib_metrics.columns).intersection(set(pf2_metrics.columns))
    )
    pf2_metrics = pf2_metrics[common_columns]
    pca_metrics = pca_metrics[common_columns]
    scib_metrics = scib_metrics[common_columns]
    all_metrics = pd.concat((scib_metrics, pf2_metrics, pca_metrics))
    all_metrics["PF2"] = ["PF2" not in method for method in all_metrics["Method"]]
    all_metrics.sort_values(by=["PF2", "Method", "Rank"], inplace=True)
    all_metrics.drop(columns=["PF2"], inplace=True)

    # reorder columns
    cols = all_metrics.columns
    all_metrics = all_metrics[
        [col for col in cols if col not in METRICS] + list(METRICS)
    ]
    all_metrics.dropna(subset=list(METRICS), inplace=True)

    all_metrics = all_metrics[
        ~all_metrics["Method"].isin(["Unintegrated", "scGen*", "scANVI*"])
    ]

    all_metrics.reset_index(drop=True, inplace=True)

    return all_metrics


def normalize_metrics(df: pd.DataFrame):
    """Min-max scale the metrics in df"""
    df = df.copy()
    scaler = MinMaxScaler()
    df[list(METRICS)] = scaler.fit_transform(df[list(METRICS)])
    return df


def filter_by_overall_score(df: pd.DataFrame):
    """Picks the best out of (unscaled, scaled) x (HVG, Full) for each method
    based on overall score (mean of min-max scaled metrics)"""
    df = normalize_metrics(df)
    df["overall"] = np.sum(df[list(METRICS)], axis=1)
    idx = df.groupby(["Method", "Dataset", "Rank"])["overall"].idxmax()
    df = df.loc[sorted(idx.values)]
    df.drop(columns=["overall"], inplace=True)
    return df


def run_nmf(df: pd.DataFrame, n_comps=2, l1_coef=0):
    """Runs nmf on the metrics df provided. Returns the scores as a df and the nmf object"""
    df = normalize_metrics(df)
    nmf = NMF(n_components=n_comps, alpha_W=l1_coef, l1_ratio=1)
    components = nmf.fit_transform(df[list(METRICS)])
    components_df = pd.DataFrame(data=components)
    non_metric_cols = [col for col in df.columns if col not in METRICS]
    components_df[non_metric_cols] = df[non_metric_cols].values
    return components_df, nmf


def get_R2(df: pd.DataFrame, nmf: sklearn.decomposition._nmf.NMF):
    scaled_data = normalize_metrics(df)[list(METRICS)]
    return 1 - nmf.reconstruction_err_ / np.linalg.norm(
        scaled_data - np.mean(scaled_data)
    )
