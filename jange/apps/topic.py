from typing import Iterable, Optional, Union, List
from jange import base, ops, stream, vis


def topic_model(
    texts: Union[stream.DataStream, Iterable],
    vectorize_op: ops.text.vectorize.SklearnBasedVectorizer,
    cleaning_ops: Optional[List[base.Operation]] = None,
    topic_modeling_op: Optional[ops.topic.TopicModelingOperation] = None,
    num_topics: int = 30,
    max_words_per_topic: int = 10,
):
    texts = texts if isinstance(texts, stream.DataStream) else stream.DataStream(texts)
    if cleaning_ops is None:
        cleaning_ops = []

    if not isinstance(vectorize_op, ops.text.vectorize.SklearnBasedVectorizer):
        raise ValueError(
            "vectorize_op should be of type ops.text.vectorize.SklearnBasedVectorizer"
            f" but got {type(vectorize_op)}"
        )

    if not topic_modeling_op:
        topic_modeling_op = ops.topic.lda(n_topics=num_topics,)

    topics_ds = texts.apply(*cleaning_ops, vectorize_op, topic_modeling_op)
    feature_names = vectorize_op.model.get_feature_names()

    return topic_modeling_op.map_topics(
        topics_ds, feature_names=feature_names, max_words_per_topic=max_words_per_topic
    )

