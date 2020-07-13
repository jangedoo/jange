import numpy as np
import pandas as pd
import plotly.express as px

from jange.stream import DataStream


def visualize(features, clusters):
    features = features.items if isinstance(features, DataStream) else features
    features = np.array(features)
    n_dim = features.shape[-1]
    if n_dim not in [2, 3]:
        raise ValueError(
            f"To visualize, the input matrix should be either 2D or 3D but got {n_dim}"
        )
    data = {}
    for axis_name, axis in zip(["x", "y", "z"], range(features.shape[-1])):
        data[axis_name] = features[:, axis]

    if isinstance(clusters, DataStream):
        context = clusters.context
        clusters = clusters.items
    else:
        context = range(len(clusters))

    data["cluster"] = clusters
    data["context"] = context
    df = pd.DataFrame(data)

    if n_dim == 2:
        fig = px.scatter(df, x="x", y="y", color="cluster", hover_data=["context"])
    else:
        fig = px.scatter_3d(
            df, x="x", y="y", z="z", color="cluster", hover_data=["context"]
        )
    return fig
