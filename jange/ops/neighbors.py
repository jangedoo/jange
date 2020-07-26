import networkx as nx
from sklearn import neighbors as sknn

from jange import base, ops, stream


class NearestNeighborsOperation(ops.base.ScikitBasedOperation):
    def __init__(self, n_neighboors: int = 10, name: str = "similar_pair") -> None:
        model = sknn.NearestNeighbors(n_neighbors=n_neighboors, metric="cosine")
        super().__init__(model=model, predict_fn_name="kneighbors", name=name)

    def _predict(self, ds):
        # nn needs access to full dataset to determine nearest neighbors
        ctx = list(ds.context)
        self.bs = len(ctx)
        for batch, context in self._get_batch(ds, ctx):
            distances, indices = self.model.kneighbors(batch)

            output = []
            for dists, indxs in zip(distances, indices):
                # get contexts for neighbors
                nbor_ctxs = [context[i] for i in indxs]
                nbors = [
                    {"context": c, "distance": d} for c, d in zip(nbor_ctxs, dists)
                ]

                output.append(nbors)

            yield output, context


class SimilarPairOperation(NearestNeighborsOperation):
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

        self.sim_threshold = sim_threshold
        model = sknn.NearestNeighbors(n_neighbors=10, metric="cosine")

        super().__init__(model=model, predict_fn_name="kneighbors", name=name)

    def _get_pairs(self, dist_indices_ds):
        def get_pair_key(id1, id2):
            return f"{id1}_{id2}"

        contexts = list(dist_indices_ds.context)
        items = list(dist_indices_ds)

        is_pair_seen = set()
        pairs = []
        for i, (data, context) in enumerate(zip(items, contexts)):
            nbor_distances = [d["distance"] for d in data]
            nbor_ctxs = [d["context"] for d in data]

            main_id = nbor_ctxs[0]
            for d, i in zip(nbor_distances[1:], nbor_ctxs[1:]):
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
        dist_indices_ds = super().run(ds)
        pairs = self._get_pairs(dist_indices_ds)

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
