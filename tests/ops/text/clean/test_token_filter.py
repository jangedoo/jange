from jange.ops.text.clean import token_filter
import pytest
import spacy
from spacy.tokens import Doc
from jange.ops.text import (
    TokenFilterOperation,
    remove_words_with_length_less_than,
    remove_stopwords,
    remove_numbers,
    remove_links,
    remove_emails,
)
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
            ["this is a ", "visit us at "],
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
    expected = ["One is ", "Hey, my number is "]
    ds = DataStream(texts)
    actual = list(map(str, ds.apply(remove_numbers())))

    assert actual == expected


def test_remove_links():
    texts = [
        "visit us at www.example.com/testing",
        "our website is http://example.com/",
    ]
    expected = ["visit us at ", "our website is "]
    ds = DataStream(texts)
    actual = list(map(str, ds.apply(remove_links())))

    assert actual == expected


def test_remove_emails():
    texts = [
        "please contact us at: info@example.com",
        "send email @ test@example.com",
    ]
    expected = ["please contact us at: ", "send email @ "]
    ds = DataStream(texts)
    actual = list(map(str, ds.apply(remove_emails())))

    assert actual == expected


def test_remove_words_with_length_less_than():
    texts = ["this is a first text", "what was that"]
    expected = ["this first text", "what that"]

    ds = DataStream(texts)
    actual = list(map(str, ds.apply(remove_words_with_length_less_than(length=4))))

    assert actual == expected
