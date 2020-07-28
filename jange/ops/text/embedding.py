"""This module contains operations for extracting word/document embeddings using a
language model.
"""
from typing import Optional

from spacy.language import Language

from jange.base import DataStream
from jange.ops.base import SpacyBasedOperation


class DocumentEmbeddingOperation(SpacyBasedOperation):
    """Operation to calculate document's vector using word-embeddings.
    Word embedding of each token are collected and averaged.

    Parameters
    ----------
    nlp : Optional[Language]
        a spacy model

    name : str
        name of this operation

    Example
    -------
    >>> ds = DataStream(["this is text 1", "this is text 2"])
    >>> vector_ds = ds.apply(DocumentEmbeddingOperation())
    >>> print(vector_ds.items)

    Attributes
    ----------
    nlp : Language
        spacy model

    name : str
        name of this operation
    """

    def __init__(
        self, nlp: Optional[Language] = None, name: str = "doc_embedding"
    ) -> None:
        super().__init__(nlp=nlp, name=name)

    def run(self, ds: DataStream) -> DataStream:
        docs_ds = self.get_docs_stream(ds)
        vecs = (d.vector for d in docs_ds)
        return DataStream(
            items=vecs, applied_ops=ds.applied_ops + [self], context=ds.context
        )


def doc_embedding(
    nlp: Optional[Language] = None, name: str = "doc_embedding"
) -> DocumentEmbeddingOperation:
    """Helper function to return DocumentEmbeddingOperation

    Parameters
    ----------
    nlp : Optional[Language]
        a spacy model

    name : str
        name of this operation

    Returns
    -------
    DocumentEmbeddingOperation
    """
    return DocumentEmbeddingOperation(nlp=nlp, name=name)
