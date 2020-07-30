"""This module contains implementation of Topic-modeling algorithms
"""

from typing import Iterable, Optional
import more_itertools

import sklearn.decomposition as skdecomp
from sklearn.base import TransformerMixin

from jange import base, ops, stream

SUPPORTED_CLASSES = [skdecomp.LatentDirichletAllocation, skdecomp.NMF]


class TopicModelingOperation(ops.base.ScikitBasedOperation):
    def __init__(
        self, model: TransformerMixin, name: Optional[str] = "topic_modeling",
    ) -> None:

        if not any(isinstance(model, cls) for cls in SUPPORTED_CLASSES):
            raise ValueError(
                f"model should be one of {SUPPORTED_CLASSES} but got {type(model)}"
            )
        super().__init__(model=model, predict_fn_name="transform", name=name)

    def _get_topic_per_item(self, raw_topics_ds: stream.DataStream):
        for topic_score, ctx in zip(raw_topics_ds, raw_topics_ds.context):
            topic_id = topic_score.argsort()[-1]
            yield topic_id, ctx

    def run(self, ds: stream.DataStream) -> stream.DataStream:
        raw_topics_scores_ds = super().run(ds)
        topics_with_ctx = self._get_topic_per_item(raw_topics_scores_ds)
        topics, ctxs = more_itertools.unzip(topics_with_ctx)
        return stream.DataStream(
            items=topics, applied_ops=ds.applied_ops + [self], context=ctxs
        )

    def map_topics(
        self,
        topics_ds: stream.DataStream,
        feature_names: Iterable[str],
        max_words_per_topic: int = 5,
    ) -> stream.DataStream:
        words_of_topics = []
        for topic_vec in self.model.components_:
            words_of_topic = []
            for feature_id in topic_vec.argsort()[-1 : -max_words_per_topic - 1 : -1]:
                words_of_topic.append(feature_names[feature_id])
            words_of_topics.append(words_of_topic)

        mapped_topics = (words_of_topics[topic_id] for topic_id in topics_ds)
        return stream.DataStream(
            mapped_topics, applied_ops=topics_ds.applied_ops, context=topics_ds.context
        )


def nmf(n_topics: int, name: Optional[str] = "nmf",) -> TopicModelingOperation:
    model = skdecomp.NMF(n_components=n_topics)
    return TopicModelingOperation(model=model, name=name,)


def lda(n_topics: int, name: Optional[str] = "lda",) -> TopicModelingOperation:
    model = skdecomp.LatentDirichletAllocation(n_components=n_topics)
    return TopicModelingOperation(model=model, name=name,)
