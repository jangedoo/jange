from unittest.mock import MagicMock

import pytest
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from jange.ops.base import ScikitBasedOperation
from jange.ops.text.encode import count, one_hot, tfidf
from jange.stream import DataStream

nlp = spacy.load("en_core_web_sm")


def test_tfidf_function_returns_valid_operation():
    max_features = 100
    binary = True  # kwargs should be passed to underlying model
    op = tfidf(max_features=max_features, binary=binary)
    assert isinstance(op, ScikitBasedOperation)
    assert isinstance(op.model, TfidfVectorizer)
    assert op.model.max_features == max_features
    assert op.model.binary == binary


def test_count_function_returns_valid_operation():
    max_features = 50
    binary = True  # kwargs should be passed to underlying model
    op = count(max_features=max_features, binary=binary)
    assert isinstance(op, ScikitBasedOperation)
    assert isinstance(op.model, CountVectorizer)
    assert op.model.max_features == max_features
    assert op.model.binary == binary


def test_one_hot_function_returns_valid_operation():
    max_features = 50
    op = one_hot(max_features=max_features)
    assert isinstance(op, ScikitBasedOperation)
    assert isinstance(op.model, CountVectorizer)
    assert op.model.max_features == max_features
    assert op.model.binary == True  # binary should be True for one_hot


@pytest.mark.parametrize(
    "op", [tfidf(), count(), tfidf(max_features=2), count(max_features=2)]
)
@pytest.mark.parametrize(
    "input, input_count",
    [
        (["this is text", "another one"], 2),
        # ([nlp.make_doc(t) for t in ["this is 1", "this is 2", "this is 3"]], 3),
    ],
)
@pytest.mark.parametrize("is_input_generator", [True, False])
def test_vectorizes_correctly(op, input, input_count, is_input_generator):
    # pytest doesn't seem to support parametrize generator
    # convert input to generator here
    if is_input_generator:
        input = (x for x in input)
    ds = DataStream(input)
    features_ds = ds.apply(op)
    assert len(op.model.vocabulary_) > 0
    features = list(features_ds)
    assert len(features) == input_count
    assert features[0].shape[-1] == len(op.model.vocabulary_)


def test_does_not_train_while_training_is_disabled():
    ds = DataStream(["this is text1", "this is text2"])
    op = ScikitBasedOperation(
        model=MagicMock(spec_set=TfidfVectorizer), predict_fn_name="transform"
    )
    op.should_train = False
    ds.apply(op)
    op.model.fit.assert_not_called()
