from unittest.mock import MagicMock

import pytest
import sklearn.cluster as skcluster

from jange import stream
from jange.ops import cluster


def test_kmeans_returns_clusteroperation():
    n_clusters = 2
    op = cluster.kmeans(n_clusters=n_clusters)
    assert isinstance(op, cluster.ClusterOperation)
    assert isinstance(op.model, skcluster.KMeans)
    assert op.model.n_clusters == n_clusters


def test_minibatch_kmeans_returns_clusteroperation():
    n_clusters = 2
    op = cluster.minibatch_kmeans(n_clusters=n_clusters)
    assert isinstance(op, cluster.ClusterOperation)
    assert isinstance(op.model, skcluster.MiniBatchKMeans)
    assert op.model.n_clusters == n_clusters


@pytest.mark.parametrize("model_class", cluster.SUPPORTED_CLASSES)
@pytest.mark.parametrize("should_train", [True, False])
def test_calls_appropriate_underlying_methods_for_training_and_prediction(
    model_class, should_train
):
    model = MagicMock(spec=model_class)
    ds = stream.DataStream([[0, 1, 2], [3, 4, 5]])

    op = cluster.ClusterOperation(model=model)
    op.should_train = should_train

    assert op.should_train == should_train

    test_return_value = [[0, 1], [3, 4]]

    if model_class in cluster.ALGORITHMS_NOT_SUPPORTING_NEW_INFERENCE:
        model.labels_ = test_return_value

        output = op.run(ds)

        model.fit.assert_called_once()
        assert list(output.items) == test_return_value

    elif model_class in cluster.ALGORITHMS_SUPPORTING_NEW_INFERENCE:
        model.predict.return_value = test_return_value

        output = op.run(ds)

        if should_train:
            model.fit.assert_called_once()
            model.predict.assert_called_once()
        else:
            model.fit.assert_not_called()
            model.predict.assert_called_once()

        assert list(output.items) == test_return_value
    else:
        raise Exception("model class not expected")
