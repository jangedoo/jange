"""This module contains commonly used dimension reduction algorithms
"""
from typing import Optional

from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.sparse import issparse

from jange.stream import DataStream
from jange.base import Operation, TrainableMixin


class DimensionReductionOperation(Operation, TrainableMixin):
    """Operation for reducing dimension of a multi-dimensional array.
    This operation is primarily used for reducing large feature space
    to 2D or 3D for easy visualization.

    Parameters
    ----------
    model : TransformerMixin
        a scikit-learn model that reduces the dimensions. Usually it will
        be PCA or TSNE.
    name : str
        name of this operation
    """

    def __init__(self, model: TransformerMixin, name: str = "dim_reduction") -> None:
        super().__init__(name=name)
        self.model: TransformerMixin = model

    def run(self, ds: DataStream) -> DataStream:
        features = ds.items
        if issparse(features):
            features = features.toarray()

        # if model has no transform method, then we fit the model
        # regardless of the state of `should_train` flag
        if not hasattr(self.model, "transform"):
            reduced_features = self.model.fit_transform(features)
        elif self.should_train:
            reduced_features = self.model.fit_transform(features)
        else:
            reduced_features = self.model.transform(features)

        return DataStream(
            reduced_features, applied_ops=ds.applied_ops + [self], context=ds.context
        )


def pca(n_dim: int = 2,):
    """DimensionReductionOperation with PCA

    Parameters
    ----------
    n_dim : int, optional
        reduce the original n-dimensional array to `n_dim` array, by default 2

    Returns
    -------
    DimensioonReductionOperation
    """
    model = PCA(n_components=n_dim)
    return DimensionReductionOperation(model, name="pca")


def tsne(n_dim: int = 2):
    """DimensionReductionOperation with TSNE

    Parameters
    ----------
    n_dim : int, optional
        reduce the original n-dimensional array to `n_dim` array, by default 2

    Returns
    -------
    DimensioonReductionOperation
    """
    model = TSNE(n_components=n_dim)
    return DimensionReductionOperation(model, name="tsne")
