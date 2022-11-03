import xarray as xa
import numpy as np
import seaborn as sns
import pandas as pd
from collections import Counter
import os
from os.path import dirname, join
from pathlib import Path

path_here = os.path.dirname(os.path.dirname(__file__))

def CoH_df(numCells: int = 100, markers=None):
    """"Reducing CoH dataframe with experiments consistent across patients into XA"""
    scCoH_DF = pd.read_csv('/opt/andrew/CoH_Flow_SC_NoMissingnessV1.csv').reset_index(drop=True).drop("Unnamed: 0", axis=1)
    status_DF = pd.read_csv('gmm/data/CoH_Patient_Status.csv').reset_index(drop=True).drop("Unnamed: 0", axis=1)
    
    #Renaming patients
    healthy = 0; bc = 0; health = []; cancer = []
    for i, stat in enumerate(status_DF["Patient"].values):
        statDF = status_DF.loc[status_DF["Patient"] == stat]
        if statDF["Status"].values == "Healthy":
            healthy += 1
            scCoH_DF = scCoH_DF.replace(stat, "Healthy-" + str(healthy))
        else: 
            bc += 1
            scCoH_DF = scCoH_DF.replace(stat,"BC-" + str(bc))
    
    if markers == "All":
        marker_dict = marker_dict_all
    elif markers == "Markers":
        marker_dict = marker_dict_surface
    else:
        marker_dict = marker_dict_stat
            
    scDF = scCoH_DF.sort_values(by="Patient")
    assert np.all(np.isfinite(scDF[marker_dict].to_numpy()))
    
    for mark in marker_dict:
        scDF = scDF[scDF[mark] < scDF[mark].quantile(0.995)]

    scDF.drop(columns=["Cell", "Time"], axis=1, inplace=True)

    # Normalization of markers
    scDF[justmark] = scDF.groupby(by=["Patient"])[justmark].transform(lambda x: x / np.mean(x, axis=0))
    scDF[marker_dict_stat] /= np.mean(scDF[marker_dict_stat], axis=0)
    
    experimentcells = scDF.groupby(by=["Treatment","Patient"]).size()

    # Choosing select number of cells
    scDF = scDF.groupby(by=["Treatment", "Patient"]).sample(n=numCells, random_state=1).reset_index(drop=True)
    scDF["Cell"] = np.tile(np.arange(1, numCells + 1), int(scDF.shape[0] / numCells))
    
    scDF.to_csv(join(path_here, "gmm/data/CoH_SC_DF_V2.csv"))
    
    return scDF

def CoH_xarray(numCells, cond, allmarkers):
    """Converting single cell CoH DF into  Xarray with defined markers and treatments"""
    if allmarkers == True:
        marker_dict = mark_var
        singleDF = pd.read_csv('/opt/andrew/CoH_SC_DF_V1.csv').reset_index(drop=True).drop(["Cell","Unnamed: 0"], axis=1)
    elif allmarkers == False:
        marker_dict = marker_dict_stat
        singleDF = pd.read_csv('/opt/andrew/CoH_SC_DF_V2.csv').reset_index(drop=True).drop(["Cell","Unnamed: 0"], axis=1)
    
    scDF = pd.DataFrame([])
    for con in cond:
        scDF = pd.concat([scDF, singleDF.loc[singleDF["Treatment"] == con]])
        
    scDF = scDF[np.append(marker_dict, ["CellType", "Treatment", "Patient"])]
    scDF = scDF.groupby(by=["Treatment", "Patient"]).sample(n=numCells, random_state=1).reset_index(drop=True)
    scDF["Cell"] = np.tile(np.arange(1, numCells + 1), int(scDF.shape[0] / numCells))
    scDF["CellType"] = scDF["CellType"].replace(cell_types_rename)
    
    # Changing to Xarray
    CoHxa = scDF.set_index(["Cell", "Treatment", "Patient"]).to_xarray()
    celltypeXA = CoHxa["CellType"]
    CoHxa = CoHxa.drop_vars(["CellType"])
    CoHxa = CoHxa[marker_dict].to_array(dim="Marker")
    npCoH = np.reshape(CoHxa.to_numpy(), (CoHxa.shape[0], CoHxa.shape[1], CoHxa.shape[2], -1, 1)).astype('float64')
    finalCoH = xa.DataArray(npCoH, dims=("Marker", "Cell", "Treatment", "Patient", "Throwaway 1"),
                            coords={"Marker": marker_dict, "Cell": np.arange(1, numCells + 1),
                                    "Treatment": cond, "Patient": scDF["Patient"].unique(), "Throwaway 1": ["Throwaway"]})

    # Final Xarray has dimensions [Marker, Cell Number, Treatment, Patient, 1]
    assert np.all(np.isfinite(finalCoH.to_numpy()))
    

    return finalCoH.sel(Patient=patients), scDF, celltypeXA

patients = ["Healthy-1", "Healthy-2", "Healthy-3", 
                "Healthy-4", "Healthy-5", "Healthy-6", "Healthy-7", "Healthy-8", "Healthy-9",
                "Healthy-10", "Healthy-11", "Healthy-12", "Healthy-13", "Healthy-14", 
                "Healthy-15", "Healthy-16", "Healthy-17", "Healthy-18", "Healthy-19", 
                "Healthy-20", "Healthy-21", "Healthy-22",
                "BC-1", "BC-2", "BC-3", "BC-4", "BC-5", "BC-6", "BC-7", "BC-8", "BC-9", 
                "BC-10", "BC-11", "BC-12", "BC-13", "BC-14" ]

cell_types = ["T", "CD16 NK", "CD8+", "CD4+", "CD4-/CD8-", "Treg", "Treg 1", "Treg 2", "Treg 3", "CD8 TEM", "CD8 TCM", "CD8 Naive", "CD8 TEMRA",
                  "CD4 TEM", "CD4 TCM", "CD4 Naive", "CD4 TEMRA", "CD20 B", "CD20 B Naive", "CD20 B Memory", "CD33 Myeloid", "Classical Monocyte", "NC Monocyte"]

cell_types_rename = {"T": 0, "CD16 NK": 1, "CD8+": 2, "CD4+": 3, "CD4-/CD8-": 4, "Treg": 5, "Treg 1": 6, "Treg 2": 7, "Treg 3": 8, "CD8 TEM": 9,
              "CD8 TCM": 10, "CD8 Naive": 11, "CD8 TEMRA": 12, "CD4 TEM": 13, "CD4 TCM": 14, "CD4 Naive": 15, "CD4 TEMRA": 16, "CD20 B": 17,
              "CD20 B Naive": 18, "CD20 B Memory": 19, "CD33 Myeloid": 20, "Classical Monocyte": 21,  "NC Monocyte": 22}

marker_dict_all = ['SSC-H', 'SSC-A', 'FSC-H', 'FSC-A', 'SSC-B-H', 'SSC-B-A', 'CD45RA',
                   'Live/Dead', 'CD4', 'CD16', 'CD8', 'pSTAT6', 'PD-L1', 'CD3', 'PD-1',
                   'CD14', 'CD33', 'CD27', 'pSTAT3', 'pSTAT1', 'pSmad1-2', 'FoxP3',
                   'pSTAT5', 'pSTAT4', 'CD20']

marker_dict_surface = ['CD45RA', 'CD4', 'CD16', 'CD8', 'pSTAT6', 'PD-L1', 'CD3', 'PD-1',
                       'CD14', 'CD33', 'CD27', 'pSTAT3', 'pSTAT1', 'pSmad1-2', 'FoxP3',
                       'pSTAT5', 'pSTAT4', 'CD20']

marker_dict_stat = ['pSTAT6', 'pSTAT3', 'pSTAT1', 'pSmad1-2', 'pSTAT5', 'pSTAT4']
mark_var =  ['pSTAT6', 'pSTAT3', 'pSTAT1', 'pSmad1-2', 'pSTAT5', 'pSTAT4', 'CD16',
             'CD33', 'PD-L1', 'CD27', 'FoxP3', 'CD14', 'CD20']
conditions = ['Untreated', 'IFNg-50ng', 'IL10-50ng', 'IL4-50ng', 'IL2-50ng', 'IL6-50ng']

justmark = list((Counter(marker_dict_surface) - Counter(marker_dict_stat)).elements())