from typing import Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer,
)
from spacy.language import Language

from jange.base import Operation, TrainableMixin
from jange.ops.base import SpacyBasedOperation
from jange.stream import DataStream


class SklearnBasedVectorizer(Operation, TrainableMixin):
    """Vectorize texts based on algorithms available in scikit-learn.
    TfidfVectorizer, CountVectorizer and HashingVectorizer can be used
    as the underlying model.

    Paramters
    ----------
    model : Union[TfidfVectorizer, CountVectorizer, HashingVectorizer]
        any of TfidfVectorizer, CountVectorizer, HashingVectorizer instance

    name : str
        name of this operation

    Example
    -------
    >>> ds = DataStream(["this is text1", "this is text2"])
    >>> op = SklearnBasedVectorizer(model=TfidfVectorizer(max_features=1000), name='tfidf')
    >>> ds.apply(op)

    Attributes
    ----------
    model : Union[TfidfVectorizer, CountVectorizer]
        underlying model for vectorizing texts

    name : str
        name of this operation
    """

    def __init__(
        self,
        model: Union[TfidfVectorizer, CountVectorizer],
        name: str = "sklearn_vectorizer",
    ) -> None:
        super().__init__(name=name)
        self.model = model

    def run(self, ds: DataStream) -> DataStream:
        x = list(ds.items)
        if not isinstance(x[0], str):
            x = list(map(str, x))
        if self.should_train:
            self.model.fit(x)

        vectors = self.model.transform(x)
        return DataStream(vectors, ds.applied_ops + [self], context=ds.context)


def tfidf(
    max_features: Optional[int] = None,
    max_df: Union[int, float] = 1.0,
    min_df: Union[int, float] = 1,
    ngram_range: Tuple[int, int] = (1, 1),
    norm: str = "l2",
    use_idf: bool = True,
    name: str = "tfidf",
    **kwargs
) -> SklearnBasedVectorizer:
    """Returns tfidf based feature vector extraction.
    Uses sklearn's TfidfVectorizer as underlying model.

    Parameters
    ----------
    max_features : Optional[int]
        If some value is provided then only top `max_features` words
        order by their count frequency are considered in the vocabulary

    max_df : Union[int, float]
        When building vocabulary, ignore terms that have document frequency higher
        than the given value. If the value is float, then it is considered as a ratio.

    min_df : Union[int, float]
        When building vocabulary, ignore terms that have document frequency less than
        the given valu. If the value is float, then it is considered as a ratio.

    ngram_range : Tuple[int, int]
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used. For example an ``ngram_range`` of ``(1, 1)`` means only
        unigrams, ``(1, 2)`` means unigrams and bigrams, and ``(2, 2)`` means
        only bigrams.

    norm : str
        Each output row will have unit norm, either:
        * 'l2': Sum of squares of vector elements is 1. The cosine
        similarity between two vectors is their dot product when l2 norm has
        been applied.
        * 'l1': Sum of absolute values of vector elements is 1.
        See :func:`preprocessing.normalize`.

    use_idf : bool
        Enable inverse-document-frequency reweighting.

    name : str
        name of this operation

    **kwargs
        Keyword parameters that will be passed to the initializer of CountVectorizer

    See https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    for details on the paramters and more examples.

    Returns
    -------
    SklearnBasedVectorizer
    """
    model = TfidfVectorizer(
        max_features=max_features, max_df=max_df, min_df=min_df, **kwargs
    )
    return SklearnBasedVectorizer(model=model, name=name)


def count(
    max_features: Optional[int] = None,
    max_df: Union[int, float] = 1.0,
    min_df: Union[int, float] = 1,
    ngram_range: Tuple[int, int] = (1, 1),
    name: Optional[str] = "count",
    **kwargs
) -> SklearnBasedVectorizer:
    """Returns count based feature vector extraction.
    Uses sklearn's CountVectorizer as underlying model.

    Parameters
    ----------
    max_features : Optional[int]
        If some value is provided then only top `max_features` words
        order by their count frequency are considered in the vocabulary

    max_df : Union[int, float]
        When building vocabulary, ignore terms that have document frequency higher
        than the given value. If the value is float, then it is considered as a ratio.

    min_df : Union[int, float]
        When building vocabulary, ignore terms that have document frequency less than
        the given valu. If the value is float, then it is considered as a ratio.

    ngram_range : Tuple[int, int]
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used. For example an ``ngram_range`` of ``(1, 1)`` means only
        unigrams, ``(1, 2)`` means unigrams and bigrams, and ``(2, 2)`` means
        only bigrams.

    name : str
        name of this operation

    **kwargs
        Keyword parameters that will be passed to the initializer of CountVectorizer

    See https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    for details on the paramters and more examples.

    Returns
    -------
    SklearnBasedVectorizer
    """
    model = CountVectorizer(
        max_features=max_features, max_df=max_df, min_df=min_df, **kwargs
    )
    return SklearnBasedVectorizer(model=model, name=name)


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
        docs = self.get_docs(ds)
        vecs = []
        for d in docs:
            if d.has_vector:
                vecs.append(d.vector)
            else:
                raise Exception(
                    "A spacy model with `vectors` capability is needed for extracting document vectors"
                )

        vecs = np.array(vecs)
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
