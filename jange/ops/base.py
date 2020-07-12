from typing import Iterable, List
from spacy.tokens import Doc
from jange.stream import DataStream


class Operation:
    def run(self, ds: DataStream) -> DataStream:
        raise NotImplementedError()

    def __call__(self, ds: DataStream) -> DataStream:
        return self.run(ds)


class TrainableMixin:
    def __init__(self) -> None:
        self.should_train = True


class SpacyUserMixin:
    def get_docs(self, ds: DataStream) -> Iterable[Doc]:
        """Returns an interable of spacy Doc from the datastream.
        If the data stream already contains spacy Docs then they
        are returned as-is otherwise the nlp object is used to 
        create spacy Docs

        Parameters
        ----------
        ds : DataStream
            input data stream

        Returns
        ------
        out : Iterable[spacy.tokens.Doc]
            an iterable of spacy's `Doc` objects
        """
        if ds.item_type != Doc:
            docs = self.nlp.pipe(ds)
        else:
            docs = ds.items

        return docs

    def discard_tokens_from_doc(self, doc: Doc, token_ids: List[int]) -> Doc:
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
        return Doc(self.nlp.vocab, words=words, spaces=spaces)
