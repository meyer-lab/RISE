"""Investigating metrics for datasets"""
from .common import (subplotLabel, getSetup, 
   plotMetricSCIB)
from ..imports.scib import  import_scib_metrics


def makeFigure():
   """Get a list of the axis objects and create a figure."""
   # Get list of axis objects
   ax, f = getSetup((18, 18), (4, 2))

   # Add subplot labels
   subplotLabel(ax)
   
   metricsDF, sheetName = import_scib_metrics()

   plotMetricSCIB(metricsDF, sheetName, ax[0:7])

   return f
