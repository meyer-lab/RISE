""" Methods for data import and normalization. """
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
import xarray as xa
from os.path import join

path_here = os.path.dirname(os.path.dirname(__file__))


def IL2_flowXA(saveXA=False):
    """Rearranges and normlizes data as an Xarray of
    dimensions of [Ligand, Dose, Time, Cell, Marker]"""
    if saveXA == True:
        flowArrow = pq.read_table("/opt/andrew/FlowDataGMM_hlog.pq")
        gVars = ["Time", "Dose", "Ligand", "Valency"]
        # Columns that should be trasformed
        tCols = ["Foxp3", "CD25", "CD45RA", "CD4"]
        transCols = tCols + ["pSTAT5"]

        # Data was measured for CD3/CD8/CD56 was not measured for non-Tregs/Thelpers
        # Also drop columns with missing values
        flowArrow = flowArrow.filter(pa.compute.is_finite(flowArrow["Foxp3"]))
        flowArrow = flowArrow.select(transCols + gVars + ["Cell Type"])
        flowDF = flowArrow.to_pandas()

        # Group and subset
        for mark in transCols:
            flowDF = flowDF[
                flowDF[mark] < flowDF[mark].quantile(0.995)
            ]  # Getting rid of outlier values
        flowDF[tCols] = flowDF.groupby(by=gVars)[tCols].transform(
            lambda x: x / np.std(x)
        )  # Dividing by std per experiement

        # Add valency to the name
        flowDF["Ligand"] = (
            flowDF["Ligand"] + "-" + flowDF["Valency"].apply(lambda x: f"{x:.0f}")
        )
        flowDF.drop(columns=["Valency"], axis=1, inplace=True)

        flowDF["pSTAT5"] /= np.std(
            flowDF["pSTAT5"]
        )  # For pSTAT5 only, dividing my std of all experiments
        flowDF.sort_values(by=["Time", "Dose", "Ligand"], inplace=True)

        # Filter out problematic ligands
        flowDF = flowDF.loc[flowDF["Time"] != 0.5]

        cellCount = flowDF.groupby(by=["Time", "Dose", "Ligand"]).size().values
        flowDF["Cell"] = np.concatenate([np.arange(int(cnt)) for cnt in cellCount])

        flowXA = flowDF.set_index(["Cell", "Time", "Dose", "Ligand"]).to_xarray()
        celltypeXA = flowXA["Cell Type"]
        flowXA = flowXA.drop_vars(["Cell Type"])
        flowXA = flowXA[transCols].to_array(dim="Marker")
        flowXA.values = np.nan_to_num(flowXA.values)
        # Final Xarray has dimensions [Ligand, Dose, Time, Cell, Marker]

        flowXA = flowXA.transpose()
        celltypeXA = celltypeXA.transpose()
        flowXA.to_netcdf(join(path_here, "data/IL2_flowXA.nc"))
        celltypeXA.to_netcdf(join(path_here, "data/IL2_celltypeXA.nc"))

    else:
        flowXA = xa.open_dataarray(join(path_here, "/opt/andrew/IL2_flowXA.nc"))
        celltypeXA = xa.open_dataarray(join(path_here, "/opt/andrew/IL2_celltypeXA.nc"))

    return flowXA, celltypeXA