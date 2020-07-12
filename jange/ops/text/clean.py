from typing import Optional, List, Dict, Tuple

import spacy
from spacy.tokens import Doc
from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher

from jange.stream import DataStream
from ..base import Operation, SpacyBasedOperation


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
        return DataStream(
            applied_ops=ds.applied_ops + [self], items=items, context=ds.context
        )

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


class ConvertToSpacyDocOperation(SpacyBasedOperation):
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
        super().__init__(nlp)

    def run(self, ds: DataStream) -> DataStream:
        docs = self.get_docs(ds)
        return DataStream(docs, applied_ops=ds.applied_ops + [self], context=ds.context)


def convert_to_spacy_doc(nlp: Optional[Language] = None) -> ConvertToSpacyDocOperation:
    """Helper function to return ConvertToSpacyDocOperation
    """
    return ConvertToSpacyDocOperation(nlp=nlp)


class LemmatizeOperation(SpacyBasedOperation):
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
        super().__init__(nlp)

    def _get_lemmatized_doc(self, doc):
        lemma_tokens = [t.lemma_ for t in doc]
        spaces = [t.whitespace_ == " " for t in doc]
        return Doc(self.nlp.vocab, words=lemma_tokens, spaces=spaces)

    def run(self, ds: DataStream):
        docs = self.get_docs(ds)
        items = map(self._get_lemmatized_doc, docs)
        return DataStream(
            applied_ops=ds.applied_ops + [self], items=items, context=ds.context
        )

    def __repr__(self):
        return f"LemmatizeOperation()"


def lemmatize(nlp: Optional[Language] = None) -> LemmatizeOperation:
    """Helper function to return LemmatizeOperation
    """
    return LemmatizeOperation(nlp)


class TokenFilterOperation(SpacyBasedOperation):
    def __init__(
        self,
        patterns: List[List[Dict]],
        nlp: Optional[Language] = None,
        keep_matching_tokens=False,
    ) -> None:
        super().__init__(nlp)
        self.keep_matching_tokens = keep_matching_tokens
        self.patterns = patterns
        self.matcher = Matcher(vocab=self.nlp.vocab, validate=True)

        for p in patterns:
            self.matcher.add("MATCHES", None, p)

    def _filter_tokens(self, matcher_output: Tuple[Doc, List[Tuple]]) -> Doc:
        doc, matches = matcher_output
        matching_token_ids = []
        for _, start, end in matches:
            for token in doc[start:end]:
                matching_token_ids.append(token.i)

        tokens_to_discard = matching_token_ids
        if self.keep_matching_tokens:
            tokens_to_discard = [t.i for t in doc if t.i not in matching_token_ids]

        return self.discard_tokens_from_doc(doc, tokens_to_discard)

    def run(self, ds: DataStream) -> DataStream:
        docs = self.get_docs(ds)
        match_results = self.matcher.pipe(docs, return_matches=True)
        new_docs = map(self._filter_tokens, match_results)
        return DataStream(
            new_docs, applied_ops=ds.applied_ops + [self], context=ds.context
        )


def token_filter(
    patterns: List[List[Dict]], keep_matching_tokens, nlp: Optional[Language] = None
) -> TokenFilterOperation:
    return TokenFilterOperation(
        patterns=patterns, nlp=nlp, keep_matching_tokens=keep_matching_tokens
    )


def remove_stopwords(
    words: List[str], nlp: Optional[Language] = None
) -> TokenFilterOperation:
    patterns = []
    for word in words:
        patterns.append([{"LOWER": word.lower()}])
    return TokenFilterOperation(patterns, nlp=nlp, keep_matching_tokens=False)


def remove_numbers(nlp: Optional[Language] = None) -> TokenFilterOperation:
    patterns = [[{"IS_DIGIT": True}]]
    return TokenFilterOperation(patterns, nlp=nlp, keep_matching_tokens=False)


def remove_links(nlp: Optional[Language] = None) -> TokenFilterOperation:
    patterns = [[{"LIKE_URL": True}]]
    return TokenFilterOperation(patterns, nlp=nlp, keep_matching_tokens=False)


def remove_emails(nlp: Optional[Language] = None) -> TokenFilterOperation:
    patterns = [[{"LIKE_EMAIL": True}]]
    return TokenFilterOperation(patterns, nlp=nlp, keep_matching_tokens=False)


def remove_words_with_length_less_than(
    length: int, nlp: Optional[Language] = None
) -> TokenFilterOperation:
    patterns = [[{"LENGTH": {"<": length}}]]
    return TokenFilterOperation(patterns, nlp=nlp, keep_matching_tokens=False)

