""" Methods for data import and normalization. """

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def smallDF(fracCells):
    # FracCells = Amount of cells per experiment
    flowDF = importflowDF()
    gVars = ["Time", "Dose", "Date", "Ligand"]
    # Columns that should be trasformed
    transCols = ["Foxp3", "CD25", "CD45RA", "CD4"]

    # Data was measured for CD3/CD8/CD56 was not measured for non-Tregs/Thelpers
    # Also drop columns with missing values
    flowDF = flowDF.dropna(subset=["Foxp3"]).dropna(axis=1)
    experimentcells = flowDF.groupby(by=gVars).size()
    flowDF[transCols] = flowDF.groupby(by=gVars)[transCols].transform(lambda x: (x - np.nanmean(x)) / np.nanstd(x))
    flowDF = flowDF.groupby(by=gVars).sample(n=fracCells).reset_index(drop=True)
    flowDF["Cell Type"] = flowDF["Cell Type"].apply(celltypetonumb)

    return flowDF, experimentcells


def celltypetonumb(typ):
    """Changing cell types to a number"""
    if typ == "None":
        return 1
    elif typ == "Treg":
        return 2
    else:  # Thelper
        return 3


def importflowDF():
    """Downloads all conditions, surface markers and cell types.
    Cells are labeled via Thelper, None, Treg, CD8 or NK"""

    monomeric = pq.read_table("/opt/andrew/FlowDataGMM_Mon_NoSub.pq")
    dimericwt = pq.read_table("/opt/andrew/FlowDataGMM_DimWT_NoSub.pq")
    monomeric = monomeric.to_pandas()
    dimericwt = dimericwt.to_pandas()
    # pd.concat([monomeric,dimericwt])

    return monomeric
