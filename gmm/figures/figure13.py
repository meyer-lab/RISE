"""
This creates Figure 13.
"""
import pandas as pd
import numpy as np
import seaborn as sns
from .common import subplotLabel, getSetup
from ..CoHimport import CoH_xarray, marker_dict_surface, mark_var
from ..regression import BC_status_plot
 
def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 6), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    cond = ['Untreated', 'IFNg-50ng', 'IL10-50ng', 'IL4-50ng', 'IL2-50ng', 'IL6-50ng']; numCell = 300
    rank = 10; n_cluster = 9 ; seed = 5
    cohXA, cohDF, celltypeXA = CoH_xarray(numCell,cond,allmarkers=True)
    
    BC_scatter(ax[0], cohDF, "pSTAT1", "IFNg-50ng")
    BC_scatter(ax[1], cohDF, "pSTAT5", "IL2-50ng")
    BC_scatter(ax[2], cohDF, "pSTAT3", "IL6-50ng")
    BC_status_plot(cohXA, rank, n_cluster, seed, ax[3])

    return f

def BC_scatter(ax, cohDF, marker, cytokine):
    """Plots the individual signals for a specific surface marker and treatment condition"""
    hist_DF = cohDF.loc[(cohDF["Treatment"] == cytokine)]
    hist_DF =  hist_DF.replace({"Patient": patient_stat})
    hist_DF = hist_DF[["Patient", marker]]
    healthyDF = hist_DF.loc[hist_DF["Patient"] == "Healthy"]
    bcDF = hist_DF.loc[hist_DF["Patient"] == "BC"]
    print("Treatment",cytokine)
    print("BC Ave: " + marker + " - ",bcDF[marker].mean())
    print("Healthy Ave: " + marker + " - ",healthyDF[marker].mean())
    sns.histplot(data=hist_DF, x=marker, hue="Patient", ax=ax)
        
patient_stat = {"Healthy-1": "Healthy", "Healthy-2": "Healthy", "Healthy-3": "Healthy", 
                "Healthy-4": "Healthy", "Healthy-5": "Healthy", "Healthy-6": "Healthy", "Healthy-7": "Healthy",
                "Healthy-8": "Healthy", "Healthy-9": "Healthy","Healthy-10": "Healthy", "Healthy-11": "Healthy",
                "Healthy-12": "Healthy", "Healthy-13": "Healthy", "Healthy-14": "Healthy", "Healthy-15": "Healthy",
                "Healthy-16": "Healthy", "Healthy-17": "Healthy", "Healthy-18": "Healthy", "Healthy-19": "Healthy", 
                "Healthy-20": "Healthy", "Healthy-21": "Healthy", "Healthy-22": "Healthy",
                "BC-1": "BC", "BC-2": "BC", "BC-3": "BC", "BC-4": "BC", "BC-5": "BC", "BC-6": "BC", "BC-7": "BC",
                "BC-8": "BC", "BC-9": "BC", "BC-10": "BC", "BC-11": "BC", "BC-12": "BC", "BC-13": "BC", "BC-14": "BC"}
    
