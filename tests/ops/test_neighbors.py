from unittest.mock import ANY

import numpy as np
import pytest

from jange import ops, stream


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_nearest_neighbors(metric):
    # create a features vector where 1st and 3rd item are in same direction
    # and are near to each other so that both cosine and euclidean dist work
    # similarly 2nd and 4th vectors are opposite in direction and far from
    # the remaining two so that they are similar to each other based on both
    # cosine and euclidean distance
    features = np.array(
        [[1.0, 1, 1], [-0.1, -0.1, -0.1], [1, 0.9, 0.9], [-0.1, -0.1, -0.2]]
    )
    ds = stream.DataStream(features, context=["a", "b", "c", "d"])
    op = ops.neighbors.NearestNeighborsOperation(n_neighbors=2, metric=metric)
    nbors_ds = ds.apply(op)
    nbors = list(nbors_ds)

    # distance does not matter as long as the items we expect to be same are
    # returned as neighbors
    assert nbors[0] == [
        {"context": "a", "distance": ANY, "item_idx": 0},
        {"context": "c", "distance": ANY, "item_idx": 2},
    ]

    assert nbors[1] == [
        {"context": "b", "distance": ANY, "item_idx": 1},
        {"context": "d", "distance": ANY, "item_idx": 3},
    ]

    assert nbors[2] == [
        {"context": "c", "distance": ANY, "item_idx": 2},
        {"context": "a", "distance": ANY, "item_idx": 0},
    ]

    assert nbors[3] == [
        {"context": "d", "distance": ANY, "item_idx": 3},
        {"context": "b", "distance": ANY, "item_idx": 1},
    ]


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_similar_pairs(metric):
    features = np.array(
        [[1.0, 1, 1], [-0.1, -0.1, -0.1], [1, 0.9, 0.9], [-0.1, -0.1, -0.2]]
    )
    # ds = stream.DataStream(features, context=[{"A": 1}, {"B": 2}, {"c": 3}, {"d": 4}])
    ds = stream.DataStream(features, context=["a", "b", "c", "d"])
    op = ops.neighbors.SimilarPairOperation(n_neighbors=2, metric=metric)
    pairs_ds = ds.apply(op)
    pairs = list(pairs_ds)

    assert sorted(pairs) == sorted([("a", "c", ANY), ("b", "d", ANY)])
