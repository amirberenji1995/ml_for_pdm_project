import sys
import os

project_root = os.path.abspath("../../")
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import seaborn as sns
from damavand.damavand.utils import z_score_scaler


def euclidean_centroid_distance(X, Y):
    """Computes Euclidean distance between the centroids of two distributions."""
    x_centroid = X.mean(axis=0)
    y_centroid = Y.mean(axis=0)
    return np.linalg.norm(x_centroid - y_centroid)


def rbf_mmd(X, Y, gamma=None):
    """Computes Maximum Mean Discrepancy (MMD) between two distributions."""
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)

    XX_dist = cdist(X, X, "sqeuclidean")
    YY_dist = cdist(Y, Y, "sqeuclidean")
    XY_dist = cdist(X, Y, "sqeuclidean")

    if gamma is None:
        combined = np.concatenate([XX_dist.ravel(), YY_dist.ravel()])
        median_dist = np.median(combined[combined > 0])
        gamma = 1.0 / (median_dist if median_dist > 0 else 1.0)

    K_XX = np.exp(-gamma * XX_dist)
    K_YY = np.exp(-gamma * YY_dist)
    K_XY = np.exp(-gamma * XY_dist)

    m, n = X.shape[0], Y.shape[0]
    mmd_squared = (
        (K_XX.sum() / (m * m)) + (K_YY.sum() / (n * n)) - 2.0 * (K_XY.sum() / (m * n))
    )

    return np.float64(np.sqrt(max(0.0, mmd_squared)))


def data_aggregator(operations: dict, key: str, axis: str):
    return pd.concat(
        [
            pd.concat(operations[key]["dataset"].data[op][axis])
            for op in operations[key]["dataset"].data.keys()
        ]
    ).reset_index(drop=True)


def similarity_calculator(
    operations: dict,
    state: str = "ALL",
    axix=["0", "1", "2"],
    normalization=False,
    metric=euclidean_centroid_distance,
):
    similarity_matrix = {}
    domain_names = list(operations.keys())

    for ax in axix:
        similarity_matrix[ax] = {}

        # Initialize the self-distances to 0.0 right away
        for domain in domain_names:
            similarity_matrix[ax][(domain, domain)] = 0.0

        # Loop through the upper triangle only
        for i, domain1 in enumerate(domain_names):
            for j in range(i + 1, len(domain_names)):
                domain2 = domain_names[j]

                # Data aggregation
                df1 = data_aggregator(operations, domain1, ax)
                df2 = data_aggregator(operations, domain2, ax)

                if state != "ALL":
                    df1 = df1[df1["state"] == state]
                    df2 = df2[df2["state"] == state]

                x1 = df1.iloc[:, :-5].to_numpy()
                x2 = df2.iloc[:, :-5].to_numpy()

                if normalization:
                    x1 = z_score_scaler(x1)
                    x2 = z_score_scaler(x2)

                # Compute the heavy metric exactly ONCE
                distance = metric(x1, x2)

                # Assign symmetrically to the results dictionary
                similarity_matrix[ax][(domain1, domain2)] = distance
                similarity_matrix[ax][(domain2, domain1)] = distance

    return similarity_matrix


def plot_triangular_similarity_heatmaps(data_dict, base_title="Domain Similarity"):
    """Converts a nested distance dictionary into clean, upper-triangular heatmaps."""
    axes = sorted(data_dict.keys())

    for axis in axes:
        axis_data = data_dict[axis]

        # Extract unique tools and structure into a DataFrame matrix
        tools = sorted(list(set([tool for pair in axis_data.keys() for tool in pair])))
        matrix = pd.DataFrame(0.0, index=tools, columns=tools)

        for (tool_a, tool_b), value in axis_data.items():
            matrix.loc[tool_a, tool_b] = float(value)

        # Generate a mask for the lower triangle
        # np.ones_like creates a matrix of True values; np.triu sets the upper triangle to False (unmasked)
        mask = np.ones_like(matrix, dtype=bool)
        mask[np.triu_indices_from(mask)] = False

        # Set up the matplotlib figure
        plt.figure(figsize=(10, 8))

        # Pass the mask into Seaborn
        sns.heatmap(
            matrix,
            mask=mask,  # This hides the redundant lower half
            annot=True,
            fmt=".1f",
            cmap="viridis_r",
            linewidths=0.5,
            cbar_kws={"label": "Distance Metric (Lower = More Similar)"},
        )

        plt.title(f"{base_title} - Axis {axis}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()
