from typing import Optional

import cytoolz
import spacy
from spacy.tokens import Doc
from spacy.language import Language

from jange.stream import DataStream
from ..base import Operation


class CaseChangeOperation(Operation):
    """Operation for changing case of the texts.

    Example
    --------
    >>> ds = DataStream(["AAA", "Bbb"])
    >>> list(ds.apply(CaseChangeOperation(mode="lower)))
    ["aaa", "bbb"]

    Attributes
    ----------
    mode : str
        one of ['lower', 'capitalize', 'upper']
    """

    def __init__(self, mode: str = "lower"):
        valid_modes = ["lower", "upper", "capitalize"]
        mode = mode.lower()
        if mode not in valid_modes:
            raise ValueError(
                f"Invalid value for mode passed."
                f" Expected one of {valid_modes} but received {mode}"
            )
        self.mode = mode

    def run(self, ds: DataStream):
        if self.mode == "upper":
            fn = str.upper
        elif self.mode == "capitalize":
            fn = str.capitalize
        else:
            fn = str.lower
        items = map(fn, ds)
        return DataStream(applied_ops=ds.applied_ops + [self], items=items)

    def __repr__(self):
        return f"CaseChangeOperation(mode='{self.mode}')"


def lowercase() -> CaseChangeOperation:
    """Helper function to create CaseChangeOperation with mode="lower"
    """
    return CaseChangeOperation(mode="lower")


def uppercase() -> CaseChangeOperation:
    """Helper function to create CaseChangeOperation with mode="upper"
    """
    return CaseChangeOperation(mode="upper")


class ConvertToSpacyDocOperation(Operation):
    """Convert a stream of texts to stream of spacy's `Doc`s.
    Once spacy processes a text, it creates an instance of `Doc`
    which contains a lot of information like part of speech, named
    entities and many others. It is usually better to convert texts
    to spacy's Doc and perform operations on them. For example, spacy
    has powerful pattern matching features which can be used.

    Any operation that expects a `nlp` object can benefit if you pass
    a stream of spacy `Doc`s instead of stream of strings. Otherwise those
    operations will independently convert the raw texts into spacy `Doc`
    everytime you call them!

    Example
    -------
    >>> ds = DataStream(["this is text 1", "this is text 2"])
    >>> op = ConvertToSpacyDocOperation(nlp=nlp)
    >>> ds.apply(op)

    Parameters
    ----------
    nlp : Optional[spacy.language.Language]
        spacy's language model or None. If None then by default
        `en_core_web_sm` spacy model is loaded

    Attributes
    ---------
    nlp : spacy.language.Language
        spacy's language model
    """

    def __init__(self, nlp: Optional[Language] = None) -> None:
        self.nlp = nlp or spacy.load("en_core_web_sm")

    def run(self, ds: DataStream) -> DataStream:
        docs = self.nlp.pipe(ds)
        return DataStream(docs, applied_ops=ds.applied_ops + [self])


def convert_to_spacy_doc(nlp: Optional[Language] = None) -> ConvertToSpacyDocOperation:
    """Helper function to return ConvertToSpacyDocOperation
    """
    return ConvertToSpacyDocOperation(nlp=nlp)


class LemmatizeOperation(Operation):
    """Perform lemmatization using spacy's language model

    Example
    -------
    >>> nlp = spacy.load("en_core_web_sm")
    >>> op = LemmatizeOperation(nlp=nlp)
    >>> ds = DataStream(["oranges are good"])
    >>> print(list(ds.apply(op))
    ["orange be good"]

    Parameters
    ----------
    nlp : Optional[spacy.language.Language]
        spacy's language model or None. If None then by default
        `en_core_web_sm` spacy model is loaded

    Attributes
    ---------
    nlp : spacy.language.Language
        spacy's language model
    """

    def __init__(self, nlp: Optional[Language]) -> None:
        self.nlp: Language = nlp or spacy.load("en_core_web_sm")

    def _get_lemmatized_doc(self, doc):
        return " ".join(t.lemma_ for t in doc)

    def run(self, ds: DataStream):
        first, items = cytoolz.peek(ds)
        if not isinstance(first, Doc):
            docs = self.nlp.pipe(items)
        else:
            docs = items
        items = map(self._get_lemmatized_doc, docs)
        return DataStream(applied_ops=ds.applied_ops + [self], items=items)

    def __repr__(self):
        return f"LemmatizeOperation()"


def lemmatize(nlp: Optional[Language] = None) -> LemmatizeOperation:
    """Helper function to return LemmatizeOperation
    """
    return LemmatizeOperation(nlp)

