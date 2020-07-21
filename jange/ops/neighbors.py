import networkx as nx
from sklearn import neighbors as sknn

from jange import base, stream


class SimilarPairOperation(base.Operation, base.TrainableMixin):
    """Finds similar pairs

    This operation uses nearest neighbors algorithms from sklearn.neighbors package
    to find similar items in a dataset and convert them into pairs. The input data
    stream should contain a numpy array or a scipy sparse matrix.

    Attributes
    ----------
    sim_threshold : float
        minimun similarity threshold that each should pair have to be considered as
        being similar

    model :
        any model from sklearn.neighbors package. default `sklearn.neighbors.NearestNeighbors`

    name : str
        name of this operation. default `similar_pair`


    Example
    -------
    >>> features_ds = stream.DataStream(np.random.uniform(size=(20, 100)))
    >>> op = SimilarPairOperation(sim_threshold=0.9)
    >>> similar_pairs = features_ds.apply(features_ds)
    """

    def __init__(
        self, sim_threshold=0.8, model=None, name: str = "similar_pair"
    ) -> None:
        super().__init__(name=name)
        self.sim_threshold = sim_threshold
        if model:
            self.model = model
        else:
            self.model = sknn.NearestNeighbors(n_neighbors=10, metric="cosine")

    def _get_pairs(self, dists, indices, context):
        def get_pair_key(id1, id2):
            return f"{id1}_{id2}"

        is_pair_seen = set()
        pairs = []
        for ds, idxs in zip(dists, indices):
            main_id = context[idxs[0]]  # instead of array index, use context
            for d, i in zip(ds[1:], idxs[1:]):
                i = context[i]  # instead of array index, use context
                doc1_id, doc2_id = sorted([main_id, i])
                pair_key = get_pair_key(doc1_id, doc2_id)
                sim = 1 - d
                if pair_key in is_pair_seen or sim < self.sim_threshold:
                    continue

                pairs.append((doc1_id, doc2_id, sim))
                is_pair_seen.add(pair_key)

        pairs = sorted(pairs, key=lambda p: p[2], reverse=True)
        return pairs

    def run(self, ds: stream.DataStream) -> stream.DataStream:
        vectors = ds.items
        if self.should_train:
            self.model.fit(vectors)

        dists, indices = self.model.kneighbors(vectors)
        pairs = self._get_pairs(dists, indices, ds.context)

        return stream.DataStream(pairs, applied_ops=ds.applied_ops + [self])


class GroupingOperation(base.Operation):
    """Operation to group a list of pairs.

    This operation is similar to clustering but instead requires
    a list of pairs. It then uses the pairs data to create a graph
    and find connected components to group the items.

    e.g. is there are pairs [("a", "b"), ("b", "c"), ("e", "f")] then the groups
    formed will be [{'a', 'b', 'c'}, {'e', 'f'}]

    The items in the DataStream should be a tuple where each tuple indicates a pair as
    follows: `<item1, item2, *other_properties>`. All other entries in the tuple except `item1`
    and `item2` will not be used by the operation and is discarded. Typically, the output of
    `ops.neighbors.SimilarPairOperation` is passed to this operation.

    Parameters
    ----------
    name : str
        name of this operation, default `grouping`

    Attributes
    ----------
    name : str
        name of this operation

    """

    def __init__(self, name: str = "grouping") -> None:
        super().__init__(name=name)

    def run(self, ds: stream.DataStream) -> stream.DataStream:
        G = nx.Graph()
        for pair in ds:
            idx1, idx2 = pair[0], pair[1]
            G.add_edge(idx1, idx2)
        groups = list(nx.connected_components(G))
        return stream.DataStream(groups, applied_ops=ds.applied_ops + [self])
