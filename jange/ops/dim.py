"""This module contains commonly used dimension reduction algorithms
"""
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.sparse import issparse
from jange.stream import DataStream
from jange.base import Operation, TrainableMixin


class DimensionReductionOperation(Operation, TrainableMixin):
    def __init__(self, model: TransformerMixin) -> None:
        super().__init__()
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
    model = PCA(n_components=n_dim)
    return DimensionReductionOperation(model)


def tsne(n_dim: int = 2):
    model = TSNE(n_components=n_dim)
    return DimensionReductionOperation(model)
