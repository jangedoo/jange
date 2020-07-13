import sklearn.cluster as skcluster
from sklearn.base import ClusterMixin

from jange.base import Operation, TrainableMixin
from jange.stream import DataStream


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
