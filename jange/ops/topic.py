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
        name: Optional[str] = "topic_modeling",
    ) -> None:
        super().__init__(name=name)
        self.model: TransformerMixin = model
        self.feature_names = feature_names
        self.max_features_per_topic = 15

    def run(self, ds: DataStream) -> DataStream:
        features = ds.items
        if self.should_train:
            components = self.model.fit_transform(features)
        else:
            components = self.model.transform(features)

        topics_per_item = []
        for i, topic_vec in enumerate(components):
            sorted_feature_index = topic_vec.argsort()[
                -1 : -self.max_features_per_topic - 1 : -1
            ]
            topics_per_item.append(
                [self.feature_names[i] for i in sorted_feature_index]
            )

        return DataStream(
            topics_per_item, applied_ops=ds.applied_ops + [self], context=ds.context
        )


def nmf(
    n_topics: int, feature_names: list, name: Optional[str] = "nmf"
) -> TopicModelingOperation:
    model = NMF(n_components=n_topics)
    return TopicModelingOperation(model=model, feature_names=feature_names, name=name)


def lda(
    n_topics: int, feature_names: list, name: Optional[str] = "lda"
) -> TopicModelingOperation:
    model = LatentDirichletAllocation(n_components=n_topics)
    return TopicModelingOperation(model=model, feature_names=feature_names, name=name)
