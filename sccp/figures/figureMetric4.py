import matplotlib.pyplot as plt
import seaborn as sns
from itertools import groupby

from .common import getSetup


from sccp.metric import assemble_df, METRICS, run_nmf


def makeFigure():
    dataset = "immune_cell_hum_mou"
    n_comps = 2

    metrics = assemble_df()
    metrics = metrics[metrics["Method"] == "PF2"]

    metrics_sub = metrics[metrics["Dataset"] == dataset]

    scores, nmf = run_nmf(metrics_sub, n_comps=n_comps)

    axes, fig = getSetup((17, 8), (1, 2))
    sns.heatmap(
        nmf.components_.T,
        yticklabels=METRICS,
        xticklabels=[f"Component {i + 1}" for i in range(n_comps)],
        annot=True,
        ax=axes[0],
    )
    axes[0].set_title(f"NMF Loadings ({dataset})")

    x_comp = 0
    y_comp = 1
    sns.scatterplot(
        x=x_comp,
        y=y_comp,
        data=scores,
        ax=axes[1],
        size="Features",
        style="Scaling",
        hue="Rank",
    )

    axes[1].set_xlabel(f"Component {x_comp + 1}")
    axes[1].set_ylabel(f"Component {y_comp + 1}")
    axes[1].set_title(f"NMF Scatterplot ({dataset})")
    axes[1].legend(title="Method", loc="best")

    fig.suptitle("Integration Metrics For PF2 Across Ranks, Features, and Scaling")

    return fig
