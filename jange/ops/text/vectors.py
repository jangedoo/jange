from sklearn.feature_extraction.text import TfidfVectorizer
from ..base import Operation, TrainableMixin
from jange.stream import DataStream


class TfIdfOperation(Operation, TrainableMixin):
    """Converts a stream of raw texts into a matrix
    of TF-IDF features

    Example
    -------
    >>> ds = DataStream(["this is text1", "this is text2"])
    >>> op = TfIdfOperation()
    >>> ds.apply(op)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.model = TfidfVectorizer(*args, **kwargs)

    def run(self, ds: DataStream) -> DataStream:
        x = list(ds)
        if self.should_train:
            self.model.fit(x)

        vectors = self.model.transform(x)
        return DataStream(vectors, ds.applied_ops + [self])


def tfidf(*args, **kwargs) -> TfIdfOperation:
    """Helper function for returning TfIdfOperation"""
    return TfIdfOperation(*args, **kwargs)
