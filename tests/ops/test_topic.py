from unittest.mock import ANY
import pytest
from jange import ops, stream


@pytest.mark.parametrize(
    "topic_op", [ops.topic.nmf(n_topics=2), ops.topic.lda(n_topics=2)]
)
def test_topic_modeling(topic_op):
    ds = stream.DataStream(["this is about playstation", "government governs",])
    topics_ds = ds.apply(ops.text.encode.count(name="vec"), topic_op)
    vec_op = topics_ds.applied_ops.find_by_name("vec")
    feature_names = vec_op.model.get_feature_names()

    topic_word_stream = topic_op.map_topics(
        topics_ds=topics_ds, feature_names=feature_names, max_words_per_topic=2,
    )

    assert list(topic_word_stream) == [[ANY, ANY], [ANY, ANY]]
