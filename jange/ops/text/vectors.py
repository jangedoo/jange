from os import name
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.language import Language
from jange.base import Operation, TrainableMixin
from jange.ops.base import SpacyBasedOperation
from jange.stream import DataStream


class FeatureVectorExtractionOperation:
    pass


class TfIdfOperation(Operation, TrainableMixin, FeatureVectorExtractionOperation):
    """Converts a stream of raw texts into a matrix
    of TF-IDF features

    Example
    -------
    >>> ds = DataStream(["this is text1", "this is text2"])
    >>> op = TfIdfOperation()
    >>> ds.apply(op)
    """

    def __init__(self, name: Optional[str] = "tfidf", *args, **kwargs) -> None:
        super().__init__(name=name)
        self.model = TfidfVectorizer(*args, **kwargs)

    def run(self, ds: DataStream) -> DataStream:
        x = list(ds.items)
        if not isinstance(x[0], str):
            x = list(map(str, x))
        if self.should_train:
            self.model.fit(x)

        vectors = self.model.transform(x)
        return DataStream(vectors, ds.applied_ops + [self], context=ds.context)


def tfidf(name: Optional[str] = "tfidf", *args, **kwargs) -> TfIdfOperation:
    """Helper function for returning TfIdfOperation"""
    return TfIdfOperation(name=name, *args, **kwargs)


class DocumentEmbeddingOperation(SpacyBasedOperation, FeatureVectorExtractionOperation):
    def __init__(
        self, nlp: Optional[Language] = None, name: Optional[str] = "doc_embedding"
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
        return DataStream(
            items=vecs, applied_ops=ds.applied_ops + [self], context=ds.context
        )


def doc_embedding(
    nlp: Optional[Language] = None, name: Optional[str] = "doc_embedding"
) -> DocumentEmbeddingOperation:
    return DocumentEmbeddingOperation(nlp=nlp, name=name)
