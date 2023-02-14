import os
from os.path import join
import numpy as np
import pandas as pd
from xarray import open_dataarray

path_here = os.path.dirname(os.path.dirname(__file__))


def CoH_xarray(saveXA=False):
    """Reducing CoH dataframe with experiments consistent across patients into XA"""
    if saveXA:
        scDF = (
            pd.read_csv("/opt/andrew/CoH_Flow_SC_NoMissingnessV1.csv")
            .reset_index(drop=True)
            .drop("Unnamed: 0", axis=1)
        )

        scDF.drop(columns=["Cell", "Time"], axis=1, inplace=True)

        status_DF = (
            pd.read_csv("sccp/data/CoH_Patient_Status.csv")
            .reset_index(drop=True)
            .drop("Unnamed: 0", axis=1)
        )

        # Renaming patients
        healthy = 0
        bc = 0
        for stat in status_DF["Patient"].values:
            statDF = status_DF.loc[status_DF["Patient"] == stat]
            if statDF["Status"].values == "Healthy":
                healthy += 1
                scDF.replace(stat, "Healthy-" + str(healthy), inplace=True)
            else:
                bc += 1
                scDF.replace(stat, "BC-" + str(bc), inplace=True)

        scDF.sort_values(by=["Treatment", "Patient"], inplace=True)
        assert np.all(np.isfinite(scDF[marker_dict_all].to_numpy()))

        # Changing to Xarray
        experimentcells = scDF.groupby(by=["Treatment", "Patient"]).size().values
        scDF["Cell"] = np.concatenate([np.arange(int(cnt)) for cnt in experimentcells])

        scDF.rename(columns={"CellType": "Cell Type"}, inplace=True)
        CoH_XA = scDF.set_index(["Cell", "Treatment", "Patient"]).to_xarray()

        celltypeXA = CoH_XA["Cell Type"]
        CoH_XA = CoH_XA.drop_vars(["Cell Type"])
        CoH_XA = CoH_XA[marker_dict_all].to_array(dim="Marker")
        
        CoH_XA = CoH_XA.transpose()
        celltypeXA = celltypeXA.transpose()

        CoH_XA.to_netcdf(join(path_here, "data/CoH_flowXA_AllMarkers.nc"))
        celltypeXA.to_netcdf(join(path_here, "data/CoH_celltypeXA_AllMarkers.nc"))
    else:
        CoH_XA = open_dataarray(
            join(path_here, "/opt/andrew/CoH_flowXA_AllMarkers.nc")
        )
        celltypeXA = open_dataarray(
            join(path_here, "/opt/andrew/CoH_celltypeXA_AllMarkers.nc")
        )
        
    # Final Xarray has dimensions [Patient, Treatment, Cell, Marker]

    return CoH_XA, celltypeXA


marker_dict_stat = ["pSTAT6", "pSTAT3", "pSTAT1", "pSmad1-2", "pSTAT5", "pSTAT4"]


marker_dict_surface = [
    "SSC-H",
    "SSC-A",
    "FSC-H",
    "FSC-A",
    "SSC-B-H",
    "SSC-B-A",
    "CD45RA",
    "Live/Dead",
    "CD4",
    "CD16",
    "CD8",
    "PD-L1",
    "CD3",
    "PD-1",
    "CD14",
    "CD33",
    "CD27",
    "FoxP3",
    "CD20",
]


marker_dict_all = marker_dict_stat + marker_dict_surface
