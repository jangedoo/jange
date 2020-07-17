"""This module contains implementation of Topic-modeling algorithms
"""

from typing import Iterable, Optional

import sklearn.decomposition as skdecomp
from sklearn.base import TransformerMixin

from jange.base import Operation, TrainableMixin, DataStream

SUPPORTED_CLASSES = [skdecomp.LatentDirichletAllocation, skdecomp.NMF]


class TopicModelingOperation(Operation, TrainableMixin):
    def __init__(
        self, model: TransformerMixin, name: Optional[str] = "topic_modeling",
    ) -> None:
        super().__init__(name=name)
        if not any(isinstance(model, cls) for cls in SUPPORTED_CLASSES):
            raise ValueError(
                f"model should be one of {SUPPORTED_CLASSES} but got {type(model)}"
            )
        self.model: TransformerMixin = model

    def run(self, ds: DataStream) -> DataStream:
        if self.should_train:
            self.model.fit(ds.items)

        features = self.model.transform(ds.items)

        topic_per_item = []
        # features.shape = [n_items, n_topics]
        # sort by score per topic in ascending order and take
        # the last one (with the highest score) for each row
        # using [:,-1] slice
        for topic_idx in features.argsort(axis=1)[:, -1]:
            topic_per_item.append(topic_idx)

        return DataStream(
            topic_per_item, applied_ops=ds.applied_ops + [self], context=ds.context
        )

    def map_topics(
        self,
        topics_ds: DataStream,
        feature_names: Iterable[str],
        max_words_per_topic: int = 5,
    ) -> DataStream:
        words_of_topics = []
        for topic_vec in self.model.components_:
            words_of_topic = []
            for feature_id in topic_vec.argsort()[-1 : -max_words_per_topic - 1 : -1]:
                words_of_topic.append(feature_names[feature_id])
            words_of_topics.append(words_of_topic)

        mapped_topics = [words_of_topics[topic_id] for topic_id in topics_ds]
        return DataStream(
            mapped_topics, applied_ops=topics_ds.applied_ops, context=topics_ds.context
        )


def nmf(n_topics: int, name: Optional[str] = "nmf",) -> TopicModelingOperation:
    model = skdecomp.NMF(n_components=n_topics)
    return TopicModelingOperation(model=model, name=name,)


def lda(n_topics: int, name: Optional[str] = "lda",) -> TopicModelingOperation:
    model = skdecomp.LatentDirichletAllocation(n_components=n_topics)
    return TopicModelingOperation(model=model, name=name,)
