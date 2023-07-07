"""Metrics used for Pf2 projections"""
import numpy as np
import pandas as pd
from .figures.common import flattenData, flattenProjs


def distDrugDF(data, ranks, Pf2s, PCs, drugs):
    """Plots normalized variance for either a variable or for a group of cells"""
    distDF = pd.DataFrame([])
    for ii, rank in enumerate(ranks):
        _, factors, projs, _ = Pf2s[ii]
        pf2All = np.concatenate(projs, axis=0)
        projDF = flattenProjs(data, projs)
        pcaAll = PCs[ii]
        for drug in drugs:
            pf2Drug = projDF.loc[projDF.Drug == drug].values[:, 0:-1]
            pcaDrug = pcaAll[projDF.Drug == drug]

            pf2Dist = len(projDF.Drug.unique()) * centroid_dist(pf2Drug) / centroid_dist(pf2All)
            pcaDist = len(projDF.Drug.unique()) * centroid_dist(pcaDrug) / centroid_dist(pcaAll)
            
            distDF = pd.concat([distDF, pd.DataFrame({"Rank": [rank], "Normalized Centroid Distance": pf2Dist, "Method": "PARAFAC2", "Drug": drug})])
            distDF = pd.concat([distDF, pd.DataFrame({"Rank": [rank], "Normalized Centroid Distance": pcaDist, "Method": "PCA", "Drug": drug})])
            
        
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
    
    for drug in projDF["Drug"].unique():
        pf2Drug = projDF.loc[projDF.Drug == drug].values[:, 0:-1]
        pcaDrug = pcaAll[projDF.Drug == drug]

        pf2Dist = len(projDF.Drug.unique()) * centroid_dist(pf2Drug) / centroid_dist(pf2All)
        pcaDist = len(projDF.Drug.unique()) * centroid_dist(pcaDrug) / centroid_dist(pcaAll)
        
        distDF = pd.concat([distDF, pd.DataFrame({"Normalized Centroid Distance": pf2Dist, "Method": "PARAFAC2", "Drug": [drug]})])
        distDF = pd.concat([distDF, pd.DataFrame({"Normalized Centroid Distance": pcaDist, "Method": "PCA", "Drug": [drug]})])
        
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
