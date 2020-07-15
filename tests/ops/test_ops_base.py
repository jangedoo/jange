from spacy.language import Language
from spacy.tokens import Doc

from jange.ops.base import SpacyBasedOperation
from jange.stream import DataStream
from jange.ops.utils import cached_spacy_model


nlp = cached_spacy_model("en_core_web_sm")


def test_get_docs_returns_spacy_docs():
    op = SpacyBasedOperation(nlp=nlp)
    text_docs = DataStream(["this is doc 1", "this is doc 2"])
    spacy_docs = DataStream([nlp.make_doc(d) for d in text_docs])

    assert all((isinstance(d, Doc) for d in op.get_docs(text_docs)))
    assert all((isinstance(d, Doc) for d in op.get_docs(spacy_docs)))


def test_discard_tokens_from_doc():
    op = SpacyBasedOperation(nlp=nlp)
    doc = nlp.make_doc("this is text")
    actual = op.discard_tokens_from_doc(doc, [0, 2])
    assert str(actual) == "is"


def test___getstate___does_not_contain_spacy_nlp_object():
    op = SpacyBasedOperation(nlp=nlp)
    assert not any((isinstance(obj, Language) for obj in op.__getstate__().values()))


def test___setstate__restores_operation():
    state = {"name": "myop", "model_path": "en_core_web_sm"}
    op = SpacyBasedOperation.__new__(SpacyBasedOperation)
    assert hasattr(op, "name") is False
    assert hasattr(op, "model_path") is False
    op.__setstate__(state)

    assert op.name == "myop"
    assert op.model_path == "en_core_web_sm"
    assert op.nlp == cached_spacy_model("en_core_web_sm")

