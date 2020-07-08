import pytest
import spacy
from spacy.tokens import Doc
from jange.ops.text import ConvertToSpacyDocOperation, convert_to_spacy_doc
from jange.stream import DataStream


@pytest.fixture
def texts():
    return ["oranges are good", "jange library rocks"]


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


def test_converts_correctly_for_stream_of_texts(texts, nlp):
    ds = DataStream(texts)
    op = ConvertToSpacyDocOperation(nlp=nlp)

    actual = list(ds.apply(op))
    assert all(isinstance(d, Doc) for d in actual)
    assert texts == list(map(str, actual))


def test_nlp_object_is_created_is_nothing_is_passed():
    op = ConvertToSpacyDocOperation(nlp=None)
    assert op.nlp is not None


def test_helper_fn_returns_valid_object():
    op = convert_to_spacy_doc()
    assert isinstance(op, ConvertToSpacyDocOperation)
