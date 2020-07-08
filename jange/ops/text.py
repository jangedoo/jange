import os
import pickle
import spacy

from jange.stream import DataStream
from .base import Operation


class CaseChangeOperation(Operation):
    def __init__(self, mode="lower"):
        self.mode = mode

    def run(self, ds: DataStream):
        if self.mode == "upper":
            fn = str.upper
        elif self.mode == "capitalize":
            fn = str.capitalize
        else:
            fn = str.lower
        items = map(fn, ds)
        return DataStream(applied_ops=ds.applied_ops + [self], items=items)

    def __repr__(self):
        return f"CaseChangeOperation(mode='{self.mode}')"


class LemmatizeOperation(Operation):
    def __init__(self, nlp) -> None:
        self.nlp = nlp or spacy.load("en_core_web_sm")

    def get_lemma_from_doc(self, doc):
        return " ".join(t.lemma_ for t in doc)

    def run(self, ds: DataStream):
        docs = self.nlp.pipe(ds)
        items = map(self.get_lemma_from_doc, docs)
        return DataStream(applied_ops=ds.applied_ops + [self], items=items)

    def __repr__(self):
        return f"LemmatizeOperation()"


def lowercase():
    return CaseChangeOperation(mode="lower")


def uppercase():
    return CaseChangeOperation(mode="upper")


def lemmatize(nlp=None):
    return LemmatizeOperation(nlp)


def tfidf(path: str, overwrite=False, *args, **kwargs):
    # if path exists then load the model and transform the input
    # otherwise create a new model, train it, save it and transform
    # the input
    # if os.path.exists(path):
    #     vec = pickle.load(open(path, "rb"))
    #     vec.transform()
    # else:
    #     vec = TfIdfVectorizer(*args, **kwargs)
    #     vec.fit()
    #     pickle.dump(open(path, "wb"), vec)
    #     vec.transform()
    pass
