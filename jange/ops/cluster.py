import sklearn.cluster as skcluster
from sklearn.base import ClusterMixin

from jange.stream import DataStream
from .base import Operation, TrainableMixin


class ClusterOperation(Operation, TrainableMixin):
    def __init__(self, model: ClusterMixin) -> None:
        super().__init__()
        self.model: ClusterMixin = model

    def run(self, ds: DataStream) -> DataStream:
        vectors = ds.items
        if self.should_train:
            clusters = self.model.fit_predict(vectors, None)
        else:
            if hasattr(self.model, "predict"):
                clusters = self.model.predict(vectors)
            else:
                clusters = self.model.labels_
        return DataStream(
            clusters, applied_ops=ds.applied_ops + [self], context=ds.context
        )


def kmeans(n_clusters: int, **kwargs) -> ClusterOperation:
    model = skcluster.KMeans(n_clusters=n_clusters, **kwargs)
    return ClusterOperation(model=model)


def minibatch_kmeans(n_clusters: int, **kwargs) -> ClusterOperation:
    model = skcluster.MiniBatchKMeans(n_clusters=n_clusters, **kwargs)
    return ClusterOperation(model=model)


class ClusterVisualizationOperation(Operation):
    def __init__(self, n_dim: int = 2) -> None:
        super().__init__()
        self.n_dim = n_dim

    def run(self, ds: DataStream) -> DataStream:
        # need the texts
        # need the vectorization operation
        # transform the texts into vectors
        # reduce the dimension
        features = None  # 2d or 3d array
        x = features[:0]
        y = features[:1]
        clusters = ds.items


def visualize(features, clusters, n_dim: int = 2):
    from sklearn.decomposition import PCA
    import plotly.express as px
    import pandas as pd

    features = features.items if isinstance(features, DataStream) else features
    reduced_features = PCA(n_components=n_dim).fit_transform(features)

    data = {}
    for axis_name, axis in zip(["x", "y", "z"], range(reduced_features.shape[-1])):
        data[axis_name] = reduced_features[:, axis]

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
