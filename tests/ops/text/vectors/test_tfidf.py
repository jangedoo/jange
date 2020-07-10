from unittest.mock import MagicMock
from jange.ops.text import TfIdfOperation, tfidf
from jange.stream import DataStream


def test_tfidf_function_returns_valid_operation():
    op = tfidf()
    assert isinstance(op, TfIdfOperation)


def test_vectorizes_correctly():
    ds = DataStream(["this is text1", "this is text2"])
    op = tfidf()
    features_ds = ds.apply(op)
    assert len(op.model.vocabulary_) == 4  # there are 4 distinct words
    assert features_ds.items.shape == (2, 4)


def test_does_not_train_while_training_is_disabled():
    ds = DataStream(["this is text1", "this is text2"])
    op = tfidf()
    op.model = MagicMock()
    op.should_train = False
    ds.apply(op)
    op.model.fit.assert_not_called()
