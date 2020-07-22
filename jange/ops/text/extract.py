from typing import Optional

from spacy.language import Language
from spacy.tokens import Doc

from jange.ops.base import SpacyBasedOperation


def _extract_tokens(doc: Doc, context):
    tokens = [t.text for t in doc]
    return tokens, context


def _extract_sentences(doc: Doc, context):
    sentences = []
    for sent in doc.sents:
        sentences.append(sent.text)

    return sentences, context


def tokens(nlp: Optional[Language] = None, name="tokens"):
    """Returns an operator for extracting tokens for each text
    in the stream. The output of this operator is a DataStream
    where each item is a list of string

    Parameters
    ----------
    nlp : Optional[Language], optional
        spacy's language model, by default None
        if it is none then `en_core_web_sm` model is used
    name : str, optional
        name of this operation, by default "tokens"

    Returns
    -------
    SpacyBasedOperation
    """
    op = SpacyBasedOperation(nlp=nlp, process_doc_fn=_extract_tokens, name=name,)
    return op


def sentences(nlp: Optional[Language] = None, name="sentences"):
    """Returns an operator for extracting sentences for each text
    in the stream. The output of this operator is a DataStream
    where each item is a list of string

    Parameters
    ----------
    nlp : Optional[Language], optional
        spacy's language model, by default None
        if it is none then `en_core_web_sm` model is used
    name : str, optional
        name of this operation, by default "sentences"

    Returns
    -------
    SpacyBasedOperation
    """
    op = SpacyBasedOperation(nlp=nlp, process_doc_fn=_extract_sentences, name=name,)
    return op
