from sklearn import neighbors as sknn

import networkx as nx
from jange import base, stream


class SimilarPairOperation(base.Operation, base.TrainableMixin):
    def __init__(
        self, sim_threshold=0.8, model=None, name: str = "similar_pair"
    ) -> None:
        super().__init__(name=name)
        self.sim_threshold = sim_threshold
        if model:
            self.model = model
        else:
            self.model = sknn.NearestNeighbors(n_neighbors=2, metric="cosine")

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
    def __init__(self, name: str = "grouping") -> None:
        super().__init__(name=name)

    def run(self, ds: stream.DataStream) -> stream.DataStream:
        G = nx.Graph()
        for idx1, idx2, _ in ds:
            G.add_edge(idx1, idx2)
        groups = list(nx.connected_components(G))
        return stream.DataStream(groups, applied_ops=ds.applied_ops + [self])
