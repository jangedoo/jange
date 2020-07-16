from unittest.mock import MagicMock
import pytest

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import spacy

from jange.ops.text import SklearnBasedVectorizer, tfidf, count
from jange.stream import DataStream


nlp = spacy.load("en_core_web_sm")


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
    "op", [tfidf(), count(), tfidf(max_features=2), count(max_features=2)]
)
@pytest.mark.parametrize(
    "input, input_count",
    [
        (["this is text", "another one"], 2),
        ([nlp.make_doc(t) for t in ["this is 1", "this is 2", "this is 3"]], 3),
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
    assert features_ds.items.shape == (input_count, len(op.model.vocabulary_))


def test_does_not_train_while_training_is_disabled():
    ds = DataStream(["this is text1", "this is text2"])
    op = SklearnBasedVectorizer(model=MagicMock())
    op.model.transform.return_value = ["does not", "matter"]
    op.should_train = False
    ds.apply(op)
    op.model.fit.assert_not_called()
