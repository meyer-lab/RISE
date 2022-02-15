
import numpy as np
import pandas as pd


def smallDF(fracCells):
    # FracCells = Amount of cells per experiment 
    flowDF = importflowDF()
    gVars = ["Time", "Dose", "Date", "Ligand"]
    # Columns that should be trasformed
    transCols = ["Foxp3", "CD25", "CD3", "CD8", "CD56", "CD45RA"]

    flowDF[transCols] = flowDF.groupby(by=gVars)[transCols].transform(lambda x: (x - np.nanmean(x)) / np.nanstd(x))
    flowDF = flowDF.groupby(by=gVars).sample(n=fracCells).reset_index()
    return flowDF


def importflowDF():
    """Downloads all conditions, surface markers and cell types.
    Cells are labeled via Thelper, None, Treg, CD8 or NK """
    return pd.read_feather('/opt/andrew/FlowDataGMM_Mon_Labeled.ftr')
