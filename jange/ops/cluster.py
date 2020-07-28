import sklearn.cluster as skcluster
from sklearn.base import ClusterMixin

from jange.ops.base import ScikitBasedOperation

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


class ClusterOperation(ScikitBasedOperation):
    """Operation for clustering. This class uses scikit-learn clustering models.

    Models under sklearn.cluster can be used as the underlying model to perform
    clustering.

    Parameters
    ----------
    model : sklearn.base.ClusterMixin
        See this module's SUPPORTED_CLASSES attribute to check what models are supported

    name : str
        name of this operation, default `cluster`

    Attributes
    ----------
    model : sklearn.base.ClusterMixin
        underlying clustering model

    name : str
        name of this operation

    Example
    -------
    >>> ds = DataStream(...)
    >>> ds.apply(ClusterOperation(model=sklearn.cluster.KMeans(3)))
    """

    def __init__(self, model: ClusterMixin, name: str = "cluster") -> None:
        if not any(isinstance(model, cls) for cls in SUPPORTED_CLASSES):
            raise ValueError(
                f"model should be one of {SUPPORTED_CLASSES} but got {type(model)}"
            )

        predict_fn_name = (
            "predict"
            if any(isinstance(model, c) for c in ALGORITHMS_SUPPORTING_NEW_INFERENCE)
            else "labels_"
        )
        super().__init__(model=model, predict_fn_name=predict_fn_name, name=name)


def kmeans(n_clusters: int, name: str = "kmeans", **kwargs) -> ClusterOperation:
    """Returns ClusterOperation with kmeans algorithm

    Parameters
    ----------
    n_clusters : int
        number of clusters to create

    name : str
        name of this operation, default `kmeans`

    kwargs :
        keyword arguments to pass to sklearn.cluster.KMeans class

    Returns
    -------
    ClusterOperation
        Operation with KMeans algorithm

    Example
    -------
    >>> op = kmeans(n_clusters=10)
    """
    model = skcluster.KMeans(n_clusters=n_clusters, **kwargs)
    return ClusterOperation(model=model, name=name)


def minibatch_kmeans(
    n_clusters: int, name: str = "minibatch_kmeans", **kwargs
) -> ClusterOperation:
    """Returns ClusterOperation with mini-batchkmeans algorithm

    Parameters
    ----------
    n_clusters : int
        number of clusters to create

    name : str
        name of this operation, default `minibatch_kmeans`

    kwargs :
        keyword arguments to pass to sklearn.cluster.MiniBatchKMeans class

    Returns
    -------
    ClusterOperation
        Operation with MiniBatchKMeans algorithm

    Example
    -------
    >>> op = minibatch_kmeans(n_clusters=10)
    """
    model = skcluster.MiniBatchKMeans(n_clusters=n_clusters, **kwargs)
    return ClusterOperation(model=model, name=name)
