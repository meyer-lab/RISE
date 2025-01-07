"""
Figure 6
"""

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind, zscore

from .common import getSetup, subplotLabel


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((7, 7), (3, 3))
    subplotLabel(ax)

    data = np.random.normal(loc=0, scale=1, size=200)
    sns.histplot(data, bins=20, ax=ax[0], color="m")

    plot_variance_by_average_expression(ax[1])

    plot_batch_effects(ax[2], ax[3], label="cell_type", palette="gnuplot2")
    plot_batch_effects(ax[4], ax[5], label="batch", palette="Dark2")

    for i in range(3):
        ax[i + 3].axis("equal")

    plot_trajectory(ax[6])
    plot_deg(ax[7])

    plot_normalization(ax[8])

    plot_pseudbulk(ax[2])

    for i in range(9):
        if i != 2:
            ax[i].locator_params(nbins=5)

    return f


def generate_synthetic_svg_data(n_genes=200, n_samples=5000):
    base_expression = np.random.normal(loc=0, scale=1, size=(n_samples, n_genes))
    low_var_mask = np.arange(n_genes) < n_genes * 0.7
    base_expression[:, low_var_mask] *= np.random.uniform(
        1.9, 4.8, size=(n_samples, np.sum(low_var_mask))
    )

    hvg_mask = np.arange(n_genes) >= n_genes * 0.7
    base_expression[:, hvg_mask] *= np.random.uniform(
        2, 5, size=(n_samples, np.sum(hvg_mask))
    )

    gene_metadata = pd.DataFrame(
        {
            "mean_expression": np.mean(base_expression, axis=0),
            "variance": np.var(base_expression, axis=0),
            "is_hvg": hvg_mask,
        }
    )

    return gene_metadata


def plot_variance_by_average_expression(ax):
    gene_metadata = generate_synthetic_svg_data()
    sns.scatterplot(
        gene_metadata,
        x="mean_expression",
        y="variance",
        hue="is_hvg",
        ax=ax,
        palette={True: "green", False: "black"},
    )


def generate_synthetic_single_cell_data(n_samples=100, n_features=2):
    """Generate synthetic single-cell data with batch effects"""
    cell_types = ["T_cell", "B_cell", "Macrophage"]
    batches = ["Batch_A", "Batch_B", "Batch_C"]
    data_list = []

    # Generate data for each combination of batch and cell type
    for batch in batches:
        for cell_type in cell_types:
            # Number of cells for this batch and cell type
            n_cells = n_samples // (len(batches) * len(cell_types))

            # Base expression for each cell type
            if cell_type == "T_cell":
                base_mean = 2
                base_std = 0.2
            elif cell_type == "B_cell":
                base_mean = 3
                base_std = 0.1
            else:  # Macrophage
                base_mean = 1
                base_std = 0.2

            # Batch-specific effect
            if batch == "Batch_A":
                batch_effect = 0.5
            elif batch == "Batch_B":
                batch_effect = 1.5
            elif batch == "Batch_C":
                batch_effect = -1

            expression = np.zeros((n_cells, n_features))
            for j in range(n_features):
                # Generate expression with gene-specific variation
                expression[:, j] = np.random.normal(
                    loc=(base_mean + batch_effect), scale=base_std, size=n_cells
                )

            # Create DataFrame for this subset
            subset_df = pd.DataFrame(
                expression, columns=[f"gene_{i}" for i in range(n_features)]
            )
            subset_df["cell_type"] = cell_type
            subset_df["batch"] = batch

            data_list.append(subset_df)

    # Combine all data
    full_data = pd.concat(data_list, ignore_index=True)

    return full_data


def perform_simple_batch_correction(data):
    """Perform simple batch correction by standardizing each gene within each batch"""
    # Create a copy of the data
    corrected_data = data.copy()

    # Perform batch correction for each gene
    for gene in [col for col in data.columns if col.startswith("gene_")]:
        # Group by batch and standardize
        corrected_data[gene] = data.groupby("batch")[gene].transform(
            lambda x: (x - x.mean()) / x.std()
        )

    return corrected_data


def plot_batch_effects(ax1, ax2, label, palette="Paired"):
    """Plot batch effects before and after correction"""
    original_df = generate_synthetic_single_cell_data()
    corrected_df = perform_simple_batch_correction(original_df)
    genes_to_plot = ["gene_0", "gene_1"]

    if label != "cell_type":
        sns.scatterplot(
            data=original_df,
            x=genes_to_plot[0],
            y=genes_to_plot[1],
            hue=label,
            palette=palette,
            ax=ax1,
        )
    sns.scatterplot(
        data=corrected_df,
        x=genes_to_plot[0],
        y=genes_to_plot[1],
        palette=palette,
        hue=label,
        ax=ax2,
    )


def generate_pseudotime_trajectory():
    """Generate pseudotime trajectory data."""
    # Parameters
    n_cells_main = 100  # number of cells in main trajectory
    n_cells_branch1 = 50  # number of cells in first branch
    n_cells_branch2 = 50  # number of cells in second branch

    # Time variable to simulate progression along the trajectory
    time_main = np.linspace(0, 1, n_cells_main)
    time_branch1 = np.linspace(
        1, 1.5, n_cells_branch1
    )  # branch 1 starting slightly after main trajectory
    time_branch2 = np.linspace(
        1, 1.5, n_cells_branch2
    )  # branch 2 starting slightly after main trajectory

    # Adjust branch 2 to have a decreasing y over x to simulate a downward trajectory

    # Main trajectory (type1)
    x_main = time_main + np.random.normal(0, 0.02, n_cells_main)
    y_main = time_main**2 + np.random.normal(0, 0.02, n_cells_main)

    # Branch 1 (type2) - continuing upward
    x_branch1 = time_branch1 + np.random.normal(0, 0.02, n_cells_branch1)
    y_branch1 = time_branch1**3 + np.random.normal(0, 0.02, n_cells_branch1)

    # Branch 2 (type3) - now with a downward trend
    x_branch2 = time_branch2 + np.random.normal(0, 0.02, n_cells_branch2)
    y_branch2 = time_branch2**-0.5 + np.random.normal(0, 0.02, n_cells_branch2)

    # Combine data into a single DataFrame
    x = np.concatenate([x_main, x_branch1, x_branch2])
    y = np.concatenate([y_main, y_branch1, y_branch2])
    cell_type = (
        ["type1"] * n_cells_main
        + ["type2"] * n_cells_branch1
        + ["type3"] * n_cells_branch2
    )
    time = np.concatenate([time_main, time_branch1, time_branch2])

    data = pd.DataFrame({"time": time, "x": x, "y": y, "cell_type": cell_type})

    return data


def plot_trajectory(ax):
    """Plot pseudotime trajectory."""
    df = generate_pseudotime_trajectory()
    sns.scatterplot(data=df, x="x", y="y", hue="cell_type", ax=ax, palette="rocket")


def generate_synthetic_deg_data():
    """Generate synthetic data for differential expression analysis."""
    n_genes = 350  # Number of genes
    n_samples = 10  # Number of samples per condition

    # Create random gene expression data
    # Assume control group has normal distribution with mean=5, std=1
    control_data = np.random.normal(loc=5, scale=1, size=(n_genes, n_samples))

    # Assume treatment group has similar distribution with some genes affected
    treatment_data = np.random.normal(loc=5, scale=1, size=(n_genes, n_samples))

    # Introduce differential expression in some genes
    # Let's say 10% of genes are differentially expressed
    n_diff_genes = int(0.1 * n_genes)
    treatment_data[:n_diff_genes] += np.random.normal(
        loc=2, scale=0.5, size=(n_diff_genes, n_samples)
    )

    # Convert to pandas DataFrames
    control_df = pd.DataFrame(
        control_data, columns=[f"Control_{i+1}" for i in range(n_samples)]
    )
    treatment_df = pd.DataFrame(
        treatment_data, columns=[f"Treatment_{i+1}" for i in range(n_samples)]
    )

    # Perform t-test for each gene to find differentially expressed genes
    p_values = []
    fold_changes = []

    for gene in range(n_genes):
        control_expression = control_df.iloc[gene, :]
        treatment_expression = treatment_df.iloc[gene, :]
        fold_change = treatment_expression.mean() - control_expression.mean()
        fold_changes.append(fold_change)
        _, p_value = ttest_ind(control_expression, treatment_expression)
        p_values.append(p_value)

    # Create results DataFrame
    results_df = pd.DataFrame(
        {
            "Gene": [f"Gene_{i+1}" for i in range(n_genes)],
            "Fold Change": fold_changes,
            "P-Value": p_values,
        }
    )

    # Add adjusted p-values using Bonferroni correction
    results_df["Adjusted P-Value"] = results_df["P-Value"] * n_genes
    results_df["Significant"] = results_df["Adjusted P-Value"] < 0.05

    return results_df


def plot_deg(ax):
    """Plot differential expression data."""
    df = generate_synthetic_deg_data()
    sns.scatterplot(
        data=df,
        x="Fold Change",
        y=-np.log10(df["P-Value"]),
        ax=ax,
        hue="Significant",
        palette={True: "red", False: "black"},
        legend="brief",
    )


def generate_synthetic_pseudobulk():
    """Generate synthetic data for pseudobulk analysis."""
    cell_types = ["type1", "type2", "type3"]
    conditions = ["condition1", "condition2", "condition3"]
    genes = 5

    # Generate random data for gene expression levels
    # We'll simulate expression levels for each cell type under each condition
    data = []
    for cell_type in cell_types:
        for condition in conditions:
            expression = np.random.rand(5)
            if cell_type == "type1":
                expression[1] += 1  # Set a fixed value for gene2
                expression[2] += 1  # Set a fixed value for gene3

            if cell_type == "type2":
                expression[0] += 1.5  # Set as fixed value for gene2
                expression[3] += 1.5  # Set a fixed value for gene3

            else:
                expression[4] += 2  # Set a fixed value for gene2

            # Create a dictionary for this subset
            subset_dict = {f"gene_{i+1}": expression[i] for i in range(genes)}
            subset_dict["cell_type"] = cell_type
            subset_dict["condition"] = condition

            data.append(subset_dict)

    # Create a DataFrame with the generated data
    df = pd.DataFrame(data)

    # Multiply values of specific genes by a value
    df.loc[df["cell_type"] == "type1", ["gene_2", "gene_3"]] *= (
        1.1  # Example: multiply gene2 and gene3 values by 2 for cell_type 'type1'
    )

    df = df.set_index(["cell_type", "condition"]).sort_index()

    return df


def plot_pseudbulk(ax):
    """Plot pseudobulk data."""
    df = generate_synthetic_pseudobulk()
    sns.heatmap(df, ax=ax, cmap="viridis", cbar_kws={"label": "Expression Level"})


def generate_norm():
    """Generate synthetic data for normalization."""
    n_cells = 200
    original_counts = np.concatenate(
        [
            np.random.poisson(lam=10, size=int(n_cells * 0.7)),  # low counts
            np.random.poisson(lam=20, size=int(n_cells * 0.3)),  # high counts
        ]
    )

    # Convert to a DataFrame for easier manipulation with Seaborn
    data = pd.DataFrame({"Counts": zscore(original_counts), "Type": "Original"})

    # Normalize by dividing by the median
    median_value = np.median(original_counts)
    normalized_counts = original_counts / median_value

    # Append normalized data to the DataFrame
    normalized_data = pd.DataFrame({"Counts": normalized_counts, "Type": "Normalized"})
    data = pd.concat([data, normalized_data])

    return data


def plot_normalization(ax):
    """Plots normalized data."""

    data = generate_norm()
    sns.histplot(data=data, x="Counts", hue="Type", bins=20, ax=ax)
