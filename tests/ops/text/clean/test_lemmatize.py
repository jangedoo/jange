import pytest
import spacy
from spacy.tokens import Doc
from jange.ops.text import LemmatizeOperation, lemmatize
from jange.stream import DataStream


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture
def texts():
    return ["oranges are good", "jange library rocks"]


@pytest.fixture
def lemmatized(nlp):
    return ["orange be good", "jange library rock"]


def test_lemmatizes_correctly_for_stream_of_texts(texts, lemmatized, nlp):
    ds = DataStream(texts)
    op = LemmatizeOperation(nlp=nlp)
    assert list(map(lambda d: d.text, ds.apply(op))) == lemmatized


def test_temmatizes_correctly_for_stream_of_spacy_docs(texts, lemmatized, nlp):
    op = LemmatizeOperation(nlp=nlp)
    docs = nlp.pipe(texts)
    ds = DataStream(docs).apply(op)
    assert list(map(lambda d: d.text, ds)) == lemmatized


def test_nlp_object_is_created_is_nothing_is_passed():
    op = LemmatizeOperation(nlp=None)
    assert op.nlp is not None


def test_helper_fn_returns_valid_object():
    op = lemmatize()
    assert isinstance(op, LemmatizeOperation)
