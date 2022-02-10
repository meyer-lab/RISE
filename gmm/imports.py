
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA

def smallDF(fracCells):

    # FracCells = Amount of cells per experiment 
    flowDF = importflowDF()
    uniquetime = flowDF.Time.unique()
    uniquedose = flowDF.Dose.unique()
    uniqueday = flowDF.Date.unique()
    uniquelig = flowDF.Ligand.unique()
    """Cells are labeled via Thelper, None, Treg, CD8 or NK """
    zflowDF = pd.DataFrame(columns=["Foxp3","CD25","CD4","CD45RA","pSTAT5","CellType"]) # Initializing matrix 
    for i in range(len(uniqueday)):
        for j in range(len(uniquedose)):
            for k in range(len(uniquetime)):
                for l in range(len(uniquelig)): 
                # Iterating through each day, time, dose, and ligand of dataset
                    flowdataDF = flowDF.loc[(flowDF.Date == uniqueday[i]) & (flowDF.Dose == uniquedose[j])
                    & (flowDF.Time == uniquetime[k]) & (flowDF.Ligand == uniquelig[l])]
                    if flowdataDF.empty == False:
                        pass
                    else: 
                        break # Data was measured for CD3/CD8/CD56 was not measured for the other cell types
                    flowdataDF = flowdataDF.dropna(subset = ['Foxp3'])
                    flowdataDF = flowdataDF.drop(columns=['CD56','CD3','CD8','Valency','index','Time','Date','Dose','Ligand'])
                    flowdataDF = flowdataDF.rename(columns = {'Cell Type':'CellType'})
                    sampleDF = flowdataDF.sample(n=fracCells) # Sample size taken of experiment
 
                    markerstype = sampleDF.columns[:-2]
                    fullcolumn = sampleDF.columns
                    for m,mark in enumerate(markerstype):
                        z_marker = stats.zscore(sampleDF[mark].values) # Zscoring dataset
                        sampleDF[mark] = sampleDF[mark].replace([sampleDF[mark].values],[z_marker])
                    
                    zflowDF = zflowDF.append(pd.DataFrame({"Foxp3":sampleDF.Foxp3.values,
                                "CD25":sampleDF.CD25.values,"CD4":sampleDF.CD4.values,"CD45RA":sampleDF.CD45RA.values,
                                "pSTAT5":sampleDF.pSTAT5.values,"CellType":sampleDF.CellType.values}));
    return zflowDF

def importflowDF():
    """Downloads all conditions, surface markers and cell types"""
    """Cells are labeled via Thelper, None, Treg, CD8 or NK """
    flowDF = pd.read_feather('/opt/andrew/FlowDataGMM_Mon_Labeled.ftr') 
    return flowDF

def import_func():
    return