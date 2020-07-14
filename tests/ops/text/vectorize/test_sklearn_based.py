from unittest.mock import MagicMock
import pytest

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from jange.ops.text import SklearnBasedVectorizer, tfidf, count
from jange.stream import DataStream


def test_tfidf_function_returns_valid_operation():
    max_features = 100
    binary = True  # kwargs should be passed to underlying model
    op = tfidf(max_features=max_features, binary=binary)
    assert isinstance(op, SklearnBasedVectorizer)
    assert isinstance(op.model, TfidfVectorizer)
    assert op.model.max_features == max_features
    assert op.model.binary == binary


def test_count_function_returns_valid_operation():
    max_features = 50
    binary = True  # kwargs should be passed to underlying model
    op = count(max_features=max_features, binary=binary)
    assert isinstance(op, SklearnBasedVectorizer)
    assert isinstance(op.model, CountVectorizer)
    assert op.model.max_features == max_features
    assert op.model.binary == binary


@pytest.mark.parametrize(
    "op,input,num_vocab,output_shape",
    [
        (tfidf(), ["this is text", "another one"], 5, (2, 5)),
        (count(), ["this is text", "another one"], 5, (2, 5)),
        (tfidf(max_features=2), ["this is text", "another one"], 2, (2, 2)),
        (count(max_features=2), ["this is text", "another one"], 2, (2, 2)),
    ],
)
def test_vectorizes_correctly(op, input, num_vocab, output_shape):
    ds = DataStream(input)
    features_ds = ds.apply(op)
    assert len(op.model.vocabulary_) == num_vocab
    assert features_ds.items.shape == output_shape


def test_does_not_train_while_training_is_disabled():
    ds = DataStream(["this is text1", "this is text2"])
    op = SklearnBasedVectorizer(model=MagicMock())
    op.should_train = False
    ds.apply(op)
    op.model.fit.assert_not_called()
