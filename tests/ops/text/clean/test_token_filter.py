import pytest
import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Doc

from jange.ops.text.clean import (
    TokenFilterOperation,
    remove_emails,
    remove_links,
    remove_numbers,
    remove_stopwords,
    remove_words_with_length_less_than,
    token_filter,
)
from jange.ops.utils import cached_spacy_model
from jange.stream import DataStream


def test_accepts_stream_of_texts():
    patterns = [[{"LOWER": "this"}]]
    op = TokenFilterOperation(patterns)
    ds = DataStream(["this is a string data stream"])

    output = list(ds.apply(op))
    assert isinstance(output[0], Doc)


def test_accepts_stream_of_spacy_docs():
    nlp = spacy.load("en_core_web_sm")
    patterns = [[{"LOWER": "this"}]]
    op = TokenFilterOperation(patterns)
    ds = DataStream(nlp.pipe(["this is a spacy doc data stream"]))

    output = list(ds.apply(op))
    assert isinstance(output[0], Doc)


def test_uses_passed_nlp_object():
    nlp = spacy.load("en_core_web_sm")
    op = TokenFilterOperation([], nlp=nlp)

    # check that nlp instance of the op is the same
    # that we passed
    assert op.nlp == nlp


@pytest.mark.parametrize(
    "input,patterns,expected",
    [
        (
            ["this is a text", "visit us at http://example.com"],
            [[{"LOWER": "text"}], [{"LIKE_URL": True}]],
            ["this is a", "visit us at"],
        ),
    ],
)
def test_token_filter_removes_matching_tokens(input, patterns, expected):
    op = TokenFilterOperation(patterns, keep_matching_tokens=False)
    ds = DataStream(input)
    actual = list(map(str, ds.apply(op)))
    assert actual == expected


@pytest.mark.parametrize(
    "input,patterns,expected",
    [
        (
            ["this is a text", "visit us at http://example.com"],
            [[{"LOWER": "text"}], [{"LIKE_URL": True}]],
            ["text", "http://example.com"],
        ),
    ],
)
def test_token_filter_keeps_matching_tokens(input, patterns, expected):
    op = TokenFilterOperation(patterns, keep_matching_tokens=True)
    ds = DataStream(input)
    actual = list(map(str, ds.apply(op)))
    assert actual == expected


@pytest.mark.parametrize(
    "patterns,nlp,keep_matching_tokens",
    [([], None, False), ([[{"LOWER": "the"}]], spacy.load("en_core_web_sm"), True)],
)
def test_token_filter_fn_returns_proper_object(patterns, nlp, keep_matching_tokens):
    op = token_filter(
        patterns=patterns, nlp=nlp, keep_matching_tokens=keep_matching_tokens
    )
    assert op.patterns == patterns
    assert op.keep_matching_tokens == keep_matching_tokens
    if nlp:
        assert op.nlp == nlp
    else:
        assert op.nlp is not None


def test_remove_stopwords():
    stopwords = ["this", "that", "an", "a"]
    texts = [
        "That is a nice car",
        "Python is a type of a snake",
        "This test should pass",
    ]
    expected = ["is nice car", "Python is type of snake", "test should pass"]
    ds = DataStream(texts)
    actual = list(map(str, ds.apply(remove_stopwords(stopwords))))

    assert actual == expected


def test_remove_numbers():
    texts = ["One is 1", "Hey, my number is 23458"]
    expected = ["One is", "Hey, my number is"]
    ds = DataStream(texts)
    actual = list(map(str, ds.apply(remove_numbers())))

    assert actual == expected


def test_remove_links():
    texts = [
        "visit us at www.example.com/testing",
        "our website is http://example.com/",
    ]
    expected = ["visit us at", "our website is"]
    ds = DataStream(texts)
    actual = list(map(str, ds.apply(remove_links())))

    assert actual == expected


def test_remove_emails():
    texts = [
        "please contact us at: info@example.com",
        "send email @ test@example.com",
    ]
    expected = ["please contact us at:", "send email @"]
    ds = DataStream(texts)
    actual = list(map(str, ds.apply(remove_emails())))

    assert actual == expected


def test_remove_words_with_length_less_than():
    texts = ["this is a first text", "what was that"]
    expected = ["this first text", "what that"]

    ds = DataStream(texts)
    actual = list(map(str, ds.apply(remove_words_with_length_less_than(length=4))))

    assert actual == expected


def test_if_every_token_is_removed_then_items_is_discarded():
    texts = ["this will be deleted", "this will not be deleted"]
    context = ["a", "b"]
    ds = DataStream(items=texts, context=context)
    op = remove_stopwords(["this", "will", "be", "deleted"])
    output_ds = ds.apply(op)

    actual_texts = list(output_ds.items)
    actual_context = list(output_ds.context)

    # check that we have only one text in the stream
    # and the context is "b"
    assert len(actual_texts) == 1
    assert len(actual_context) == 1
    assert actual_context[0] == "b"


def test___getstate___does_not_contain_spacy_nlp_or_matcher_object():
    op = TokenFilterOperation(patterns=[])
    assert not any(
        (
            isinstance(obj, Language) or isinstance(obj, Matcher)
            for obj in op.__getstate__().values()
        )
    )


def test___setstate__restores_operation():
    state = {
        "name": "myop",
        "model_path": "en_core_web_sm",
        "keep_matching_tokens": False,
        "patterns": [[{"LOWER": "token"}]],
    }
    op = TokenFilterOperation.__new__(TokenFilterOperation)
    assert hasattr(op, "name") is False
    assert hasattr(op, "model_path") is False
    assert hasattr(op, "keep_matching_tokens") is False
    assert hasattr(op, "patterns") is False

    op.__setstate__(state)

    assert op.name == "myop"
    assert op.model_path == "en_core_web_sm"
    assert op.nlp == cached_spacy_model("en_core_web_sm")
    assert op.patterns == [[{"LOWER": "token"}]]
    assert isinstance(op.matcher, Matcher)
