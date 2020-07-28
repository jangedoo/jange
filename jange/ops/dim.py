"""This module contains commonly used dimension reduction algorithms
"""
import sklearn.decomposition as skd
import sklearn.manifold as skm
from sklearn.base import TransformerMixin

from jange.ops.base import ScikitBasedOperation

# some algorithms do not support predicting on new samples
# and needs retraining all the time. separate those algorithms
ALGORITHMS_SUPPORTING_NEW_INFERENCE = [
    skm.Isomap,
    skm.LocallyLinearEmbedding,
    skd.DictionaryLearning,
    skd.FactorAnalysis,
    skd.FastICA,
    skd.IncrementalPCA,
    skd.KernelPCA,
    skd.LatentDirichletAllocation,
    skd.MiniBatchDictionaryLearning,
    skd.MiniBatchSparsePCA,
    skd.NMF,
    skd.PCA,
    skd.SparsePCA,
    skd.SparseCoder,
]
ALGORITHMS_NOT_SUPPORTING_NEW_INFERENCE = [skm.MDS, skm.SpectralEmbedding, skm.TSNE]

SUPPORTED_CLASSES = (
    ALGORITHMS_SUPPORTING_NEW_INFERENCE + ALGORITHMS_NOT_SUPPORTING_NEW_INFERENCE
)


class DimensionReductionOperation(ScikitBasedOperation):
    """Operation for reducing dimension of a multi-dimensional array.
    This operation is primarily used for reducing large feature space
    to 2D or 3D for easy visualization.

    Parameters
    ----------
    model : TransformerMixin
        a scikit-learn model that reduces the dimensions. Usually it will
        be PCA or TSNE. See `SUPPORTED_CLASSES` for all scikit-learn models
        that are supported
    name : str
        name of this operation
    """

    def __init__(self, model: TransformerMixin, name: str = "dim_reduction") -> None:

        if not any(isinstance(model, cls) for cls in SUPPORTED_CLASSES):
            raise ValueError(
                f"model should be one of {SUPPORTED_CLASSES} but got {type(model)}"
            )
        predict_fn_name = (
            "transform"
            if any(isinstance(model, c) for c in ALGORITHMS_SUPPORTING_NEW_INFERENCE)
            else "embedding_"
        )
        super().__init__(model=model, predict_fn_name=predict_fn_name, name=name)


def pca(n_dim: int = 2,) -> DimensionReductionOperation:
    """DimensionReductionOperation with PCA

    Parameters
    ----------
    n_dim : int, optional
        reduce the original n-dimensional array to `n_dim` array, by default 2

    Returns
    -------
    DimensionReductionOperation
    """
    model = skd.PCA(n_components=n_dim)
    return DimensionReductionOperation(model, name="pca")


def tsne(n_dim: int = 2) -> DimensionReductionOperation:
    """DimensionReductionOperation with TSNE

    Parameters
    ----------
    n_dim : int, optional
        reduce the original n-dimensional array to `n_dim` array, by default 2

    Returns
    -------
    DimensionReductionOperation
    """
    model = skm.TSNE(n_components=n_dim)
    return DimensionReductionOperation(model, name="tsne")
