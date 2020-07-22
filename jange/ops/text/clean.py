"""This module contains several text cleaning operations
"""
from typing import Dict, List, Optional, Tuple, Union

import more_itertools
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Doc

from jange.stream import DataStream

from ..base import Operation, SpacyBasedOperation


class EmptyTextError(Exception):
    pass


class CaseChangeOperation(Operation):
    """Operation for changing case of the texts.

    Parameters
    ----------
    mode : str
        one of `lower`, `upper` or `capitalize`

    name : str
        name of this operation

    Example
    --------
    >>> ds = DataStream(["AAA", "Bbb"])
    >>> list(ds.apply(CaseChangeOperation(mode="lower)))
    ["aaa", "bbb"]

    Attributes
    ----------
    mode : str
        one of ['lower', 'capitalize', 'upper']

    name : str
        name of this operation
    """

    def __init__(self, mode: str = "lower", name: str = "case_change"):
        super().__init__(name=name)
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


def lowercase(name="lowercase") -> CaseChangeOperation:
    """Helper function to create CaseChangeOperation with mode="lower"
    """
    return CaseChangeOperation(mode="lower", name=name)


def uppercase(name="uppercase") -> CaseChangeOperation:
    """Helper function to create CaseChangeOperation with mode="upper"
    """
    return CaseChangeOperation(mode="upper", name=name)


def _lemmatize(doc, ctx):
    lemma_tokens = [t.lemma_ for t in doc]
    return " ".join(lemma_tokens), ctx


def lemmatize(nlp: Optional[Language] = None, name="lemmatize") -> SpacyBasedOperation:
    """Helper function to return SpacyBasedOperation for lemmatizing.
    This operation returns a DataStream where each item is a string after
    being lemmatized.

    Parameters
    ----------
    nlp : Optional[spacy.language.Language]
        spacy's language model or None. If None then by default
        `en_core_web_sm` spacy model is loaded

    name : Optional[str]
        name of this operation

    Returns
    -------
    out : SpacyBasedOperation
    """
    return SpacyBasedOperation(nlp=nlp, process_doc_fn=_lemmatize, name=name,)


class TokenFilterOperation(SpacyBasedOperation):
    """Operation for filtering individual tokens.

    Spacy's token pattern matching is used for matching various
    tokens in the document. Any tokens matching the filter can
    either be discarded or kept while discarding the non matching ones.

    Parameters
    ----------
    patterns : List[List[Dict]]
        a list of patterns where each pattern is a List[Dict]. The patterns
        are passed to spacy's Token Matcher.
        see https://spacy.io/usage/rule-based-matching for more details
        on how to define patterns.

    nlp : Optional[spacy.language.Language]
        spacy's language model or None. If None then by default
        `en_core_web_sm` spacy model is loaded

    keep_matching_tokens: bool
        if true then any non-matching tokens are discarded from the document (e.g. extracting only nouns)
        if false then any matching tokens are discarded (e.g. stopword removal)

    name : Optional[str]
        name of this operation

    Example
    -------
    >>> nlp = spacy.load("en_core_web_sm")
    >>> # define patterns to match [a, an, the] tokens
    >>> patterns = [
        [{"LOWER": "a"}],
        [{"LOWER": "an"}],
        [{"LOWER": "the"}]
    ]
    >>> # define the token filter operation to match the patterns and discard them
    >>> op = TokenFilterOperation(patterns=patterns, nlp=nlp, keep_matching_tokens=False)
    >>> ds = DataStream(["that is an orange"])
    >>> print(list(ds.apply(op))
    ["that is orange"]

    See https://spacy.io/usage/rule-based-matching#adding-patterns-attributes for more details
    on what token patterns can be used.

    Attributes
    ---------
    nlp : spacy.language.Language
        spacy's language model

    keep_matching_tokens : bool
        whether to discard the tokens matched by the filter from the document
        or to keep them

    patterns : List[List[Dict]]
        patterns to pass to spacy's Matcher

    name : str
        name of this operation

    """

    def __init__(
        self,
        patterns: List[List[Dict]],
        nlp: Optional[Language] = None,
        keep_matching_tokens=False,
        name: Optional[str] = "token_filter",
    ) -> None:
        super().__init__(nlp, name=name)
        self.keep_matching_tokens = keep_matching_tokens
        self.patterns = patterns
        self.matcher = self._get_matcher(self.nlp, self.patterns)

    def _get_matcher(self, nlp, patterns):
        matcher = Matcher(vocab=nlp.vocab, validate=True)

        for p in patterns:
            matcher.add("MATCHES", None, p)

        return matcher

    def _discard_tokens_from_doc(self, doc: Doc, token_ids: List[int]) -> Doc:
        """Returns a new document after discarding the tokens

        Parameters
        ----------
        doc : spacy.tokens.Doc
            orignal document
        token_ids : List[int]
            a list of index of tokens to discard

        Returns
        -------
        out : spacy.tokens.Doc
            a new document which does not contain the tokens specified
        """
        tokens = [t for t in doc if t.i not in token_ids]
        words = [t.text for t in tokens]
        spaces = [t.whitespace_ == " " for t in tokens]
        spaces[-1] = False
        return Doc(self.nlp.vocab, words=words, spaces=spaces)

    def _filter_tokens(self, matcher_output: Tuple[Doc, List[Tuple]]) -> Doc:
        ((doc, matches), context) = matcher_output
        matching_token_ids = []
        for _, start, end in matches:
            for token in doc[start:end]:
                matching_token_ids.append(token.i)

        tokens_to_discard = matching_token_ids
        if self.keep_matching_tokens:
            tokens_to_discard = [t.i for t in doc if t.i not in matching_token_ids]
        # if we have to discard all tokens in the document
        # then throw an exception
        if len(tokens_to_discard) == len(doc):
            raise EmptyTextError
        else:
            return self._discard_tokens_from_doc(doc, tokens_to_discard).text, context

    def run(self, ds: DataStream) -> DataStream:
        docs_ds = self.get_docs_stream(ds)
        docs = zip(docs_ds, docs_ds.context)
        # match results is a tuple ((doc, matches), context)
        match_results = self.matcher.pipe(docs, return_matches=True, as_tuples=True)
        new_docs_with_context = more_itertools.map_except(
            self._filter_tokens, match_results, EmptyTextError
        )
        new_docs, context = more_itertools.unzip(new_docs_with_context)
        return DataStream(
            new_docs, applied_ops=ds.applied_ops + [self], context=context
        )

    def __getstate__(self):
        state = super().__getstate__()
        del state["matcher"]
        return state

    def __setstate__(self, state: dict):
        super().__setstate__(state)
        self.matcher = self._get_matcher(self.nlp, self.patterns)

    def __repr__(self) -> str:
        patterns = (
            self.patterns
            if len(self.patterns) < 10
            else f"{self.patterns[:10]}... and others"
        )
        return f"TokenFilterOperation(patterns={patterns}, keep_matching_tokens={self.keep_matching_tokens}, name={self.name})"


def token_filter(
    patterns: List[List[Dict]],
    keep_matching_tokens,
    nlp: Optional[Language] = None,
    name: Optional[str] = "token_filter",
) -> TokenFilterOperation:
    """Helper function to create TokenFilterOperation

    Parameters
    ----------
    patterns : List[List[Dict]]
        a list of patterns where each pattern is a List[Dict]. The patterns
        are passed to spacy's Token Matcher.
        see https://spacy.io/usage/rule-based-matching for more details
        on how to define patterns.

    nlp : Optional[spacy.language.Language]
        spacy's language model or None. If None then by default
        `en_core_web_sm` spacy model is loaded

    keep_matching_tokens: bool
        if true then any non-matching tokens are discarded from the document (e.g. extracting only nouns)
        if false then any matching tokens are discarded (e.g. stopword removal)

    name : Optional[str]
        name of this operation

    Returns
    -------
    TokenFilterOperation
    """
    return TokenFilterOperation(
        patterns=patterns,
        nlp=nlp,
        keep_matching_tokens=keep_matching_tokens,
        name=name,
    )


def filter_pos(
    pos_tags: Union[str, List[str]],
    keep_matching_tokens: bool = False,
    nlp: Optional[Language] = None,
    name: Optional[str] = "filter_pos",
) -> TokenFilterOperation:
    """TokenFilterOperation to filter tokens based on Part of Speech

    Parameters
    ----------
    pos_tags : Union[str, List[str]]
        a single POS tag or a list of POS tags to search for.
        See https://spacy.io/api/annotation#pos-tagging for more details on
        what tags can be used. These depend on the language model used.

    keep_matching_tokens : bool
        if true then tokens having the given part of speech are kept and
        others are discarded from the text. Otherwise, tokens not having
        the given part of speech tags are kept

    nlp : Optional[spacy.language.Language]
        spacy's language model or None. If None then by default
        `en_core_web_sm` spacy model is loaded

    name : Optional[str]
        name of this operation

    Returns
    -------
    TokenFilterOperation

    Example
    -------
    >>> ds = stream.DataStream(["Python is a programming language"])
    >>> print(list(ds.apply(ops.text.filter_pos("NOUN", keep_matching_tokens=True))))
    [programming language]

    """
    patterns = []
    if not isinstance(pos_tags, (list, tuple)):
        pos_tags = [pos_tags]
    for tag in pos_tags:
        patterns.append([{"POS": tag}])
    return TokenFilterOperation(
        patterns, nlp=nlp, keep_matching_tokens=keep_matching_tokens, name=name
    )


def remove_stopwords(
    words: List[str] = None,
    nlp: Optional[Language] = None,
    name: Optional[str] = "remove_stopwords",
) -> TokenFilterOperation:
    """TokenFilterOperation to remove stopwords

    Parameters
    ----------
    words : List[str]
        a list of words to remove from the text

    nlp : Optional[spacy.language.Language]
        spacy's language model or None. If None then by default
        `en_core_web_sm` spacy model is loaded

    name : Optional[str]
        name of this operation

    Returns
    -------
    TokenFilterOperation

    Example
    -------
    >>> ds = stream.DataStream(["Python is a programming language"])
    >>> print(list(ds.apply(ops.text.remove_stopwords())))
    [Python programming language]
    >>> print(list(ds.apply(ops.text.remove_stopwords(words=["programming]))))
    [Python is a language]
    """
    patterns = []
    if words:
        for word in words:
            patterns.append([{"LOWER": word.lower()}])
    else:
        patterns.append([{"IS_STOP": True}])

    return TokenFilterOperation(
        patterns, nlp=nlp, keep_matching_tokens=False, name=name
    )


def remove_numbers(
    nlp: Optional[Language] = None, name: Optional[str] = "remove_numbers"
) -> TokenFilterOperation:
    """TokenFilterOperation to remove numbers

    Parameters
    ----------
    nlp : Optional[spacy.language.Language]
        spacy's language model or None. If None then by default
        `en_core_web_sm` spacy model is loaded

    name : Optional[str]
        name of this operation

    Returns
    -------
    TokenFilterOperation
    """
    patterns = [[{"IS_DIGIT": True}]]
    return TokenFilterOperation(
        patterns, nlp=nlp, keep_matching_tokens=False, name=name
    )


def remove_links(
    nlp: Optional[Language] = None, name: Optional[str] = "remove_links"
) -> TokenFilterOperation:
    """TokenFilterOperation to remove hyperlinks

    Parameters
    ----------
    nlp : Optional[spacy.language.Language]
        spacy's language model or None. If None then by default
        `en_core_web_sm` spacy model is loaded

    name : Optional[str]
        name of this operation

    Returns
    -------
    TokenFilterOperation
    """
    patterns = [[{"LIKE_URL": True}]]
    return TokenFilterOperation(
        patterns, nlp=nlp, keep_matching_tokens=False, name=name
    )


def remove_emails(
    nlp: Optional[Language] = None, name: Optional[str] = "remove_emails"
) -> TokenFilterOperation:
    """TokenFilterOperation to remove emails

    Parameters
    ----------
    nlp : Optional[spacy.language.Language]
        spacy's language model or None. If None then by default
        `en_core_web_sm` spacy model is loaded

    name : Optional[str]
        name of this operation

    Returns
    -------
    TokenFilterOperation
    """
    patterns = [[{"LIKE_EMAIL": True}]]
    return TokenFilterOperation(
        patterns, nlp=nlp, keep_matching_tokens=False, name=name
    )


def remove_words_with_length_less_than(
    length: int,
    nlp: Optional[Language] = None,
    name: Optional[str] = "remove_words_with_length_less_than",
) -> TokenFilterOperation:
    """TokenFilterOperation to remove tokens that have fewer characters
    than specified

    Parameters
    ----------
    length : int
        atleast this many characters should be in the token, otherwise
        it is discarded

    nlp : Optional[spacy.language.Language]
        spacy's language model or None. If None then by default
        `en_core_web_sm` spacy model is loaded

    name : Optional[str]
        name of this operation

    Returns
    -------
    TokenFilterOperation
    """
    patterns = [[{"LENGTH": {"<": length}}]]
    return TokenFilterOperation(
        patterns, nlp=nlp, keep_matching_tokens=False, name=name,
    )
