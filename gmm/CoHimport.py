
import xarray as xa
import numpy as np
import seaborn as sns
import pandas as pd
from collections import Counter

marker_dict_all = ['SSC-H', 'SSC-A', 'FSC-H', 'FSC-A', 'SSC-B-H', 'SSC-B-A', 'CD45RA',
                        'Live/Dead', 'CD4', 'CD16', 'CD8', 'pSTAT6', 'PD-L1', 'CD3', 'PD-1',
                        'CD14', 'CD33', 'CD27', 'pSTAT3', 'pSTAT1', 'pSmad1-2', 'FoxP3',
                        'pSTAT5', 'pSTAT4', 'CD20']
        
marker_dict_surface = ['CD45RA', 'CD4', 'CD16', 'CD8', 'pSTAT6', 'PD-L1', 'CD3', 'PD-1',
                    'CD14', 'CD33', 'CD27', 'pSTAT3', 'pSTAT1', 'pSmad1-2', 'FoxP3',
                    'pSTAT5', 'pSTAT4', 'CD20']

marker_dict_stat = ['pSTAT6','pSTAT3', 'pSTAT1', 'pSmad1-2', 'pSTAT5', 'pSTAT4']

justmark = list((Counter(marker_dict_surface)-Counter(marker_dict_stat)).elements())

def CoH_xarray(numCells: int = 100, markers=None):
    """"Reducing CoH dataframe with experiments consistent across patients into XA"""
    scCoH_DF = pd.read_csv('/opt/andrew/CoH_Flow_SC.csv').reset_index(drop=True).drop("Unnamed: 0",axis=1)
    
    if markers == "All":
        marker_dict = marker_dict_all   
    elif markers == "Markers":
        marker_dict = marker_dict_surface
    else:
        marker_dict = marker_dict_stat
        
    conditions = ['Untreated','IFNg-50ng','IL10-50ng','IL4-50ng','IL2-50ng','IL6-50ng']
    variables = ["Time", "Treatment", "Patient"]
    
    assert np.all(np.isfinite(scCoH_DF[marker_dict].to_numpy()))
    
    scDF = pd.DataFrame([])
    for cond in conditions:
        partialDF = scCoH_DF.loc[(scCoH_DF["Treatment"] == cond) & (scCoH_DF["Time"] == "15min")]
        scDF = pd.concat([scDF,partialDF])
    
    for mark in marker_dict:
        scDF = scDF[scDF[mark] < scDF[mark].quantile(0.995)] 
        
    scDF.sort_values(by=variables, inplace=True)
    scDF.drop(columns=["Cell","CellType","Time"], axis=1, inplace=True)
    
    # Normalization of markers
    scDF[justmark] = scDF.groupby(by=["Patient"])[justmark].transform(lambda x: x/np.mean(x, axis=0))
    scDF[marker_dict_stat] /= np.mean(scDF[marker_dict_stat],axis=0) 
    
    scDF = scDF.groupby(by=["Treatment","Patient"]).sample(n=numCells, random_state=1).reset_index(drop=True)
    scDF["Cell"] = np.tile(np.arange(1, numCells + 1), int(scDF.shape[0] / numCells))
    CoHxa = scDF.set_index(["Cell", "Treatment", "Patient"]).to_xarray()
    CoHxa = CoHxa[marker_dict].to_array(dim="Marker")

    npCoH = np.reshape(CoHxa.to_numpy(), (CoHxa.shape[0], CoHxa.shape[1], CoHxa.shape[2], -1, 1)).astype('float64')
    finalCoH = xa.DataArray(npCoH, dims=("Marker", "Cell", "Treatment", "Patient", "Throwaway 1"),
        coords={"Marker": marker_dict, "Cell": np.arange(1, numCells + 1),
            "Treatment": conditions, "Patient": scDF["Patient"].unique(), "Throwaway 1": ["Throwaway"]})
    
    # Final Xarray has dimensions [Marker, Cell Number, Treatment, Patient, 1]
    assert np.all(np.isfinite(finalCoH.to_numpy()))
    
    return finalCoH
    
    