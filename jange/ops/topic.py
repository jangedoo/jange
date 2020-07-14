"""This module contains implementation of Topic-modeling algorithms
"""

from typing import Optional

from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.base import TransformerMixin

from jange.base import Operation, TrainableMixin, DataStream


class TopicModelingOperation(Operation, TrainableMixin):
    def __init__(
        self,
        model: TransformerMixin,
        feature_names: list,
        max_words_per_topic: Optional[int] = 10,
        name: Optional[str] = "topic_modeling",
    ) -> None:
        super().__init__(name=name)
        self.model: TransformerMixin = model
        self.feature_names = feature_names
        self.max_words_per_topic = max_words_per_topic
        self.topics = []

    def run(self, ds: DataStream) -> DataStream:
        if self.should_train:
            self.model.fit(ds.items)

            for i, topic_vec in enumerate(self.model.components_):
                words_of_topic = []
                for feature_id in topic_vec.argsort()[
                    -1 : -self.max_words_per_topic - 1 : -1
                ]:
                    words_of_topic.append(self.feature_names[feature_id])
                self.topics.append(words_of_topic)

        features = self.model.transform(ds.items)

        topic_per_item = []
        # features.shape = [n_items, n_topics]
        # sort by score per topic in ascending order and take
        # the last one (with the heighest score) for each row
        # using [:,-1] slice
        for topic_idx in features.argsort(axis=1)[:, -1]:
            topic_per_item.append(self.topics[topic_idx])

        return DataStream(
            topic_per_item, applied_ops=ds.applied_ops + [self], context=ds.context
        )


def nmf(
    n_topics: int,
    feature_names: list,
    max_words_per_topic: Optional[int] = 10,
    name: Optional[str] = "nmf",
) -> TopicModelingOperation:
    model = NMF(n_components=n_topics)
    return TopicModelingOperation(
        model=model,
        feature_names=feature_names,
        max_words_per_topic=max_words_per_topic,
        name=name,
    )


def lda(
    n_topics: int,
    feature_names: list,
    max_words_per_topic: Optional[str] = 10,
    name: Optional[str] = "lda",
) -> TopicModelingOperation:
    model = LatentDirichletAllocation(n_components=n_topics)
    return TopicModelingOperation(
        model=model,
        feature_names=feature_names,
        max_words_per_topic=max_words_per_topic,
        name=name,
    )
