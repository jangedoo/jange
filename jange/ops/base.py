from typing import Iterable, List, Optional
from spacy.language import Language
from spacy.tokens import Doc
from jange.base import Operation
from jange.stream import DataStream
from jange import config
from .utils import cached_spacy_model


class SpacyBasedOperation(Operation):
    def __init__(self, nlp: Optional[Language]) -> None:
        super().__init__()
        self.nlp = nlp or cached_spacy_model(config.DEFAULT_SPACY_MODEL)
        self.model_path = self.nlp.path

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
            docs = ds

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

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["nlp"]
        return state

    def __setstate__(self, state: dict):
        self.__dict__.update(state)
        nlp = cached_spacy_model(state["model_path"])
        self.nlp = nlp
