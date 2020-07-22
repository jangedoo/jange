import pytest
import spacy

from jange.ops.text.clean import lemmatize
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
    op = lemmatize(nlp=nlp)
    assert list(ds.apply(op)) == lemmatized


def test_lemmatizes_correctly_for_stream_of_spacy_docs(texts, lemmatized, nlp):
    op = lemmatize(nlp=nlp)
    docs = nlp.pipe(texts)
    assert list(DataStream(docs).apply(op)) == lemmatized


def test_nlp_object_is_created_is_nothing_is_passed():
    op = lemmatize(nlp=None)
    assert op.nlp is not None
