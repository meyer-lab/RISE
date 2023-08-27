"""Metrics used for Pf2 projections"""
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler

from .figures.common import flattenData, flattenProjs

DATA_DIR = Path(__file__).parent / "data"

def distDrugDF(data, ranks, Pf2s, PCs, conds):
    """Plots normalized variance for either a variable or for a group of cells"""
    distDF = pd.DataFrame([])
    for ii, rank in enumerate(ranks):
        _, factors, projs, _ = Pf2s[ii]
        pf2All = np.concatenate(projs, axis=0)
        projDF = flattenProjs(data, projs)
        pcaAll = PCs[ii]
        for cond in conds:
            pf2Cond = projDF.loc[projDF["Condition"] == cond].values[:, 0:-1]
            pcaCond = pcaAll[projDF["Condition"] == cond]

            pf2Dist = len(projDF["Condition"].unique()) * centroid_dist(pf2Cond) / centroid_dist(pf2All)
            pcaDist = len(projDF["Condition"].unique()) * centroid_dist(pcaCond) / centroid_dist(pcaAll)
            
            distDF = pd.concat([distDF, pd.DataFrame({"Rank": [rank], "Normalized Centroid Distance": pf2Dist, "Method": "PARAFAC2", "Condition": cond})])
            distDF = pd.concat([distDF, pd.DataFrame({"Rank": [rank], "Normalized Centroid Distance": pcaDist, "Method": "PCA", "Condition": cond})])
            
        
    distDF = distDF.reset_index(drop=True)
            
    return distDF

def distGeneDF(data, ranks, Pf2s, PCs, markers):
    """Plots normalized variance for either a variable or for a group of cells"""
    distDF = pd.DataFrame([])
    for ii, rank in enumerate(ranks):
        _, factors, projs, _ = Pf2s[ii]
        dataDF = flattenData(data)
        projDF = flattenProjs(data, projs)
        pf2All = np.concatenate(projs, axis=0)
        pcaAll = PCs[ii]
        for marker in markers:
            dataDF[marker + " status"] = "Marker Negative"
            dataDF.loc[dataDF[marker] > 0.03, marker + " status"] = "Marker Positive"
        
            pf2Marker = projDF.loc[
                dataDF[marker + " status"] == "Marker Positive"
            ].values[:, 0:-1]      
            pcaMarker = pcaAll[dataDF[marker + " status"] == "Marker Positive"]
            
            pf2Dist = centroid_dist(pf2Marker) / centroid_dist(pf2All)
            pcaDist = centroid_dist(pcaMarker) / centroid_dist(pcaAll)
            
            distDF = pd.concat([distDF, pd.DataFrame({"Rank": [rank], "Normalized Centroid Distance": pf2Dist, "Method": "PARAFAC2",  "Marker": marker})])
            distDF = pd.concat([distDF, pd.DataFrame({"Rank": [rank], "Normalized Centroid Distance": pcaDist, "Method": "PCA",  "Marker": marker})])
            
        
    distDF = distDF.reset_index(drop=True)
            
    return distDF


def distAllDrugDF(data, Pf2s, PCs):
    """Plots normalized variance for either a variable or for a group of cells"""
    distDF = pd.DataFrame([])
    
    factors = Pf2s[1]
    projs = Pf2s[2]
    pf2All = np.concatenate(projs, axis=0)
    projDF = flattenProjs(data, projs)
    pcaAll = PCs
    
    for cond in projDF["Condition"].unique():
        pf2Cond = projDF.loc[projDF["Condition"] == cond].values[:, 0:-1]
        pcaCond = pcaAll[projDF["Condition"] == cond]

        pf2Dist = len(projDF["Condition"].unique()) * centroid_dist(pf2Cond) / centroid_dist(pf2All)
        pcaDist = len(projDF["Condition"].unique()) * centroid_dist(pcaCond) / centroid_dist(pcaAll)
        
        distDF = pd.concat([distDF, pd.DataFrame({"Normalized Centroid Distance": pf2Dist, "Method": "PARAFAC2", "Condition": [cond]})])
        distDF = pd.concat([distDF, pd.DataFrame({"Normalized Centroid Distance": pcaDist, "Method": "PCA", "Condition": [cond]})])
        
    distDF = distDF.reset_index(drop=True)
            
    return distDF


def distAllGeneDF(data, Pf2s, PCs):
    """Plots normalized variance for either a variable or for a group of cells"""
    distDF = pd.DataFrame([])
    
    factors = Pf2s[1]
    projs = Pf2s[2]
    dataDF = flattenData(data)
    projDF = flattenProjs(data, projs)
    pf2All = np.concatenate(projs, axis=0)
    pcaAll = PCs
    
    markers = [item for value in marker_genes.values() for item in (value if isinstance(value, list) else [value])]

    datDFcopy = dataDF.copy()
    for marker in markers:
        if marker in datDFcopy.columns:
            dataDF.loc[dataDF[marker] > 0.03, marker + " status"] = "Marker Positive"
            
    for marker in markers:
        if marker in datDFcopy.columns:
            pf2Gene = projDF.loc[dataDF[marker + " status"] == "Marker Positive"].values[:, 0: -1]
            pcaGene = pcaAll[dataDF[marker + " status"] == "Marker Positive"]

            pf2Dist = centroid_dist(pf2Gene) / centroid_dist(pf2All)
            pcaDist = centroid_dist(pcaGene) / centroid_dist(pcaAll)
            distDF = pd.concat([distDF, pd.DataFrame({"Marker": [marker], "Normalized Centroid Distance": pf2Dist, "Method": "PARAFAC2"})])
            distDF = pd.concat([distDF, pd.DataFrame({"Marker": [marker], "Normalized Centroid Distance": pcaDist, "Method": "PCA"})])

    
    distDF = distDF.reset_index(drop=True)
    
    return distDF

def centroid_dist(points):
    """Calculates mean distance between clust points and all others"""
    centroid = np.mean(points, axis=0)
    dist_vecs = points - np.reshape(centroid, (1, -1))
    row_norms = np.linalg.norm(np.array(dist_vecs, dtype=np.float64), axis=0)
    return np.mean(np.square(row_norms))

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
    df["Output"] = [method.split('/')[-1].split('_')[1] for method in df["Method"]]
    df["Features"] = [
        {"full_feature": "FULL", "hvg": "HVG"}[method.split("/")[4]]
        for method in df["Method"]
    ]
    df["Scaling"] = [method.split("/")[3] for method in df["Method"]]
    df = df[~df["Method"].str.contains("unintegrated")]
    df["Rank"] = df["Rank"].astype(int)
    df["Method"] = [method.split('/')[-1].split('_')[0].upper() for method in df["Method"]]
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
    pf2_metrics = preprocess_pf2_pca_metrics(pd.read_excel(DATA_DIR / "Pf2-metrics.xlsx"))
    pca_metrics = preprocess_pf2_pca_metrics(pd.read_excel(DATA_DIR / "pca-metrics.xlsx").rename(columns={"Unnamed: 0": "Method"}))
    scib_metrics_separate = pd.read_excel(DATA_DIR / "ScibMetrics.xlsx", sheet_name=None)
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
    common_columns = list(set(scib_metrics.columns).intersection(set(pf2_metrics.columns)))
    pf2_metrics = pf2_metrics[common_columns]
    pca_metrics = pca_metrics[common_columns]
    scib_metrics = scib_metrics[common_columns]
    all_metrics = pd.concat((scib_metrics, pf2_metrics, pca_metrics))
    all_metrics["PF2"] = ["PF2" not in method for method in all_metrics["Method"]]
    all_metrics.sort_values(by=["PF2", "Method", "Rank"], inplace=True)
    all_metrics.drop(columns=["PF2"], inplace=True)

    # reorder columns
    cols = all_metrics.columns
    all_metrics = all_metrics[[col for col in cols if col not in METRICS] + list(METRICS)]
    all_metrics.dropna(subset=list(METRICS), inplace=True)

    all_metrics = all_metrics[~all_metrics["Method"].isin(["Unintegrated", "scGen*", "scANVI*"])]

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
    return 1 - nmf.reconstruction_err_ / np.linalg.norm(scaled_data - np.mean(scaled_data))
