"""Investigating metrics for datasets"""
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotMetrics import plotMetricNormSCIB, plotMetricSCIB
from ..imports.scib import import_scib_metrics, normalize_scib


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((30, 18), (4, 4))

    # Add subplot labels
    subplotLabel(ax)

    metricsDF, metricsName, sheetName = import_scib_metrics()
    plotMetricSCIB(metricsDF, sheetName, ax[0:7])

    metricsNormDF, metricsName, sheetName = normalize_scib(
        metricsDF, metricsName, sheetName
    )
    plotMetricNormSCIB(metricsNormDF, sheetName, ax[7:14])

    return f
