"""This module contains several text encoding algorithms including binary or one-hot encoding,
count based and tf-idf
"""
from typing import Optional, Tuple, Union

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from jange.ops.base import ScikitBasedOperation


def tfidf(
    max_features: Optional[int] = None,
    max_df: Union[int, float] = 1.0,
    min_df: Union[int, float] = 1,
    ngram_range: Tuple[int, int] = (1, 1),
    norm: str = "l2",
    use_idf: bool = True,
    name: str = "tfidf",
    **kwargs,
) -> ScikitBasedOperation:
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

    use_idf : bool
        Enable inverse-document-frequency reweighting.

    name : str
        name of this operation

    **kwargs
        Keyword parameters that will be passed to the initializer of CountVectorizer

    Returns
    -------
    SklearnBasedEncodeOperation

    See https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    for details on the paramters and more examples.
    """
    model = TfidfVectorizer(
        max_features=max_features,
        max_df=max_df,
        min_df=min_df,
        ngram_range=ngram_range,
        norm=norm,
        use_idf=use_idf,
        **kwargs,
    )
    return ScikitBasedOperation(model=model, predict_fn_name="transform", name=name)


def count(
    max_features: Optional[int] = None,
    max_df: Union[int, float] = 1.0,
    min_df: Union[int, float] = 1,
    ngram_range: Tuple[int, int] = (1, 1),
    name: Optional[str] = "count",
    **kwargs,
) -> ScikitBasedOperation:
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

    Returns
    -------
    SklearnBasedEncodeOperation

    See https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    for details on the paramters and more examples.
    """
    model = CountVectorizer(
        max_features=max_features,
        max_df=max_df,
        min_df=min_df,
        ngram_range=ngram_range,
        **kwargs,
    )
    return ScikitBasedOperation(model=model, predict_fn_name="transform", name=name)


def one_hot(
    max_features: Optional[int] = None,
    max_df: Union[int, float] = 1.0,
    min_df: Union[int, float] = 1,
    ngram_range: Tuple[int, int] = (1, 1),
    name: Optional[str] = "one_hot",
    **kwargs,
) -> ScikitBasedOperation:
    """Returns operation for performing one hot encoding of texts.

    Uses sklearn.feature_extraction.text.CountVectorizer class with binary=True mode

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

    Returns
    -------
    SklearnBasedEncodeOperation

    See https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    for details on the paramters and more examples.
    """
    model = CountVectorizer(
        max_features=max_features,
        max_df=max_df,
        min_df=min_df,
        ngram_range=ngram_range,
        binary=True,
        **kwargs,
    )
    return ScikitBasedOperation(model=model, predict_fn_name="transform", name=name)
