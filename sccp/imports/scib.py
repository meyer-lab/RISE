import anndata
import pandas as pd
from .scRNA import tensorFy
import numpy as np

def import_scib_data(dataname):
    """Immune cell import from Theis lab: scib pipeline"""
    if dataname == "ImmuneHuman":
        annData = anndata.read_h5ad("/opt/andrew/scib/Immune_ALL_human.h5ad")
        annData.obs.rename(columns={"batch": "Condition"}, inplace=True)
        celltypes = annData.obs["final_annotation"].values.to_numpy()
    elif dataname == "ImmuneHumanMouse":
        annData = anndata.read_h5ad("/opt/andrew/scib/Immune_ALL_hum_mou.h5ad")
        annData.obs.rename(columns={"batch": "Condition"}, inplace=True)
        celltypes = annData.obs["final_annotation"].values.to_numpy()
    elif dataname == "Stimulation1":
        annData = anndata.read_h5ad("/opt/andrew/scib/sim1_1_norm.h5ad")
        annData.obs.rename(columns={"Batch": "Condition"}, inplace=True)
        celltypes = annData.obs["Group"].values.to_numpy()
    elif dataname == "Stimulation2":
        annData = anndata.read_h5ad("/opt/andrew/scib/sim2_norm.h5ad")
        annData.obs.rename(columns={"SubBatch": "Condition"}, inplace=True)
        celltypes = annData.obs["Group"].values.to_numpy()
    elif dataname == "Pancreas":
        annData = anndata.read_h5ad("/opt/andrew/scib/human_pancreas_norm_complexBatch.h5ad")
        annData.obs.rename(columns={"tech": "Condition"}, inplace=True)
        celltypes = annData.obs["celltype"].values.to_numpy()
        
    return tensorFy(annData, "Condition"), celltypes

def import_scib_metrics():
    """Imports all metric outputs from SCIB including Pf2 for all datasets"""
    xls = pd.ExcelFile("sccp/data/ScibMetrics.xlsx")
    
    # metrics = ["PCR batch", "Batch ASW", "graph iLISI", "graph connectivity", 
    #            "kBET", "NMI cluster/label", "ARI cluster/label", 
    #            "Cell type ASW", "isolated label F1", 
    #            "isolated label silhouette", "graph cLISI"]

    metrics = ["PCR batch", "Batch ASW", "graph iLISI",	
               "graph connectivity", "kBET"]
    
    sheet_name=["immune_cell_hum","immune_cell_hum_mou", "simulations_1_1", 
                "simulations_2", "pancreas", "mouse_brain_atac_genes_small", 
                "mouse_brain_atac_genes_large"]
    
    df = pd.DataFrame([])
    for i in range(len(sheet_name)):
        df = pd.concat([df, pd.read_excel(xls, sheet_name[i])])
        
    df = df.loc[(df["Features"] == "FULL") & (df["Scaling"] == "unscaled")]
    # No specification large and small atac excel sheet; 
    # assumed they are unscaled and full features
    
    df = df[np.append(metrics, ["Dataset","Method"])]

    df = pd.melt(df, id_vars=["Dataset", "Method"], value_vars=metrics).rename(
            columns={"variable": "Metric", "value": "Value"})
        
    return df.dropna(), sheet_name