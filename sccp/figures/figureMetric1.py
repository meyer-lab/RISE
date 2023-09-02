import matplotlib.pyplot as plt
import seaborn as sns
from itertools import groupby

from sccp.figures.common import getSetup
from sccp.metric import assemble_df, filter_by_overall_score, METRICS, run_nmf

dataset = "immune_cell_hum_mou"
n_comps = 2

def makeFigure():
    metrics = assemble_df()
    metrics = metrics[metrics["Method"] != "PCA"]

    metrics_sub = metrics[metrics["Dataset"] == dataset]
    methods = [key for key, _ in groupby(metrics_sub["Method"])]

    metrics_sub = filter_by_overall_score(metrics_sub)
    scores, nmf = run_nmf(metrics_sub, n_comps=n_comps)

    axes, fig = getSetup((17, 8), (1, 2))
    sns.heatmap(nmf.components_.T, yticklabels=METRICS, xticklabels=[f"Component {i + 1}" for i in range(n_comps)], annot=True, ax=axes[0])
    axes[0].set_title(f'NMF Loadings ({dataset})')

    x_comp = 0
    y_comp = 1
    palette = sns.color_palette("Paired")
    palette = (palette[:10] + [palette[-1]])[:len(methods)]
    sns.scatterplot(x=x_comp, y=y_comp, hue="Method", data=scores, palette=palette, s=50, ax=axes[1])
    scores_sub = scores[scores["Method"] == "PF2"]
    idx = methods.index("PF2")
    color = sns.color_palette(palette)[idx]
    for i in range(len(scores_sub) - 1):
        axes[1].plot([scores_sub.iloc[i][x_comp], scores_sub.iloc[i+1][x_comp]], [scores_sub.iloc[i][y_comp], scores_sub.iloc[i+1][y_comp]], color=color)
    axes[1].set_xlabel(f"Component {x_comp + 1}")
    axes[1].set_ylabel(f"Component {y_comp + 1}")
    axes[1].set_title(f'NMF Scatterplot ({dataset})')
    axes[1].legend(title="Method", loc='best')

    return fig

