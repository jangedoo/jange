from typing import Optional
import sklearn.cluster as skcluster
from sklearn.base import ClusterMixin

from jange.base import Operation, TrainableMixin
from jange.stream import DataStream

# some algorithms do not support predicting on new samples
# and needs retraining all the time. separate those algorithms
ALGORITHMS_SUPPORTING_NEW_INFERENCE = [
    skcluster.KMeans,
    skcluster.MiniBatchKMeans,
    skcluster.AffinityPropagation,
    skcluster.Birch,
    skcluster.MeanShift,
]
ALGORITHMS_NOT_SUPPORTING_NEW_INFERENCE = [
    skcluster.AgglomerativeClustering,
    skcluster.DBSCAN,
    skcluster.OPTICS,
    skcluster.SpectralClustering,
]

SUPPORTED_CLASSES = (
    ALGORITHMS_SUPPORTING_NEW_INFERENCE + ALGORITHMS_NOT_SUPPORTING_NEW_INFERENCE
)


class ClusterOperation(Operation, TrainableMixin):
    def __init__(self, model: ClusterMixin, name: Optional[str] = "cluster") -> None:
        super().__init__(name=name)
        if not any(isinstance(model, cls) for cls in SUPPORTED_CLASSES):
            raise ValueError(
                f"model should be one of {SUPPORTED_CLASSES} but got {type(model)}"
            )

        self.model: ClusterMixin = model

    def run(self, ds: DataStream) -> DataStream:
        vectors = ds.items

        # if model does not have predict method then
        # regardless of training state, call fit_predict
        if not hasattr(self.model, "predict"):
            clusters = self.model.fit_predict(vectors)
        else:
            if self.should_train:
                self.model.fit(vectors)

            clusters = self.model.predict(vectors)
        return DataStream(
            clusters, applied_ops=ds.applied_ops + [self], context=ds.context
        )


def kmeans(n_clusters: int, **kwargs) -> ClusterOperation:
    model = skcluster.KMeans(n_clusters=n_clusters, **kwargs)
    return ClusterOperation(model=model, name="kmeans")


def minibatch_kmeans(n_clusters: int, **kwargs) -> ClusterOperation:
    model = skcluster.MiniBatchKMeans(n_clusters=n_clusters, **kwargs)
    return ClusterOperation(model=model, name="minibatch_kmeans")
