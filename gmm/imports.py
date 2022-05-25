""" Methods for data import and normalization. """

import numpy as np
import pyarrow.parquet as pq


def smallDF(numCells: int):
    """Creates Xarray of a specific # of experiments
    Zscores all markers per experiment but pSTAT5 normalized over all experiments
    Outputs amount of experiments and cell types as an Xarray"""
    # numCells = Amount of cells per experiment
    flowDF = pq.read_table("/opt/andrew/FlowDataGMM_Mon_NoSub.pq")
    flowDF = flowDF.to_pandas()
    gVars = ["Time", "Dose", "Ligand"]
    # Columns that should be trasformed
    tCols = ["Foxp3", "CD25", "CD45RA", "CD4"]
    transCols = tCols + ["pSTAT5"]

    flowDF["Ligand"] = flowDF["Ligand"] + "-" + flowDF["Valency"].astype(int).astype(str)
    flowDF.drop(columns=["Valency", "index", "Date"], axis=1, inplace=True)

    # Data was measured for CD3/CD8/CD56 was not measured for non-Tregs/Thelpers
    # Also drop columns with missing values
    flowDF = flowDF.dropna(subset=["Foxp3"]).dropna(axis=1)
    experimentcells = flowDF.groupby(by=gVars).size()
    flowDF[tCols] = flowDF.groupby(by=gVars)[tCols].transform(lambda x: x / np.std(x))  # Dividing by std per experiement
    for mark in transCols:
        flowDF = flowDF[flowDF[mark] < flowDF[mark].quantile(0.995)]  # Getting rid of outlier values
    flowDF = flowDF.groupby(by=gVars).sample(n=numCells).reset_index(drop=True)
    flowDF["Cell Type"] = flowDF["Cell Type"].replace({"None": 1, "Treg": 2, "Thelper": 3})
    flowDF["pSTAT5"] /= np.std(flowDF["pSTAT5"])  # For pSTAT5 only, dividing my std of all experiments
    flowDF.sort_values(by=gVars, inplace=True)

    flowDF["Cell"] = np.tile(np.arange(1, numCells + 1), int(flowDF.shape[0] / numCells))
    flowDF = flowDF.set_index(["Cell", "Time", "Dose", "Ligand"]).to_xarray()
    cell_type = flowDF["Cell Type"]
    flowDF = flowDF.drop_vars(["Cell Type"])
    flowDF = flowDF[transCols].to_array(dim="Marker")
    # Final Xarray has dimensions [Marker, Cell Number, Time, Dose, Ligand]

    return flowDF, (experimentcells, cell_type)
