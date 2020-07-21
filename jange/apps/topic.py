from typing import Iterable, List, Optional, Union

from jange import base, ops, stream


def topic_model(
    texts: Union[stream.DataStream, Iterable],
    num_topics: int = 30,
    max_words_per_topic: int = 10,
    vectorize_op: ops.text.encode.SklearnBasedEncodeOperation = None,
    cleaning_ops: Optional[List[base.Operation]] = None,
    topic_modeling_op: Optional[ops.topic.TopicModelingOperation] = None,
):
    texts = texts if isinstance(texts, stream.DataStream) else stream.DataStream(texts)
    if cleaning_ops is None:
        cleaning_ops = []

    if vectorize_op and not isinstance(
        vectorize_op, ops.text.encode.SklearnBasedEncodeOperation
    ):
        raise ValueError(
            "vectorize_op should be of type ops.text.encode.SklearnBasedEncodeOperation"
            f" but got {type(vectorize_op)}"
        )
    elif vectorize_op is None:
        vectorize_op = ops.text.encode.tfidf(max_features=15000, max_df=0.98, min_df=2)

    if not topic_modeling_op:
        topic_modeling_op = ops.topic.lda(n_topics=num_topics,)

    topics_ds = texts.apply(*cleaning_ops, vectorize_op, topic_modeling_op)
    feature_names = vectorize_op.model.get_feature_names()

    return topic_modeling_op.map_topics(
        topics_ds, feature_names=feature_names, max_words_per_topic=max_words_per_topic
    )
