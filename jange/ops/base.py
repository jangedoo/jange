from typing import Callable, Optional

import more_itertools
from spacy.language import Language
from spacy.tokens import Doc

from jange import config
from jange.base import Operation
from jange.stream import DataStream

from .utils import cached_spacy_model


class SpacyModelPicklerMixin:
    def __getstate__(self):
        state = self.__dict__.copy()
        model_path = state["nlp"].path
        state["model_path"] = model_path
        del state["nlp"]
        return state

    def __setstate__(self, state: dict):
        self.__dict__.update(state)
        nlp = cached_spacy_model(state["model_path"])
        self.nlp = nlp


def _noop_process_doc_fn(doc, ctx):
    return doc, ctx


class SpacyBasedOperation(Operation, SpacyModelPicklerMixin):
    def __init__(
        self,
        nlp: Optional[Language] = None,
        process_doc_fn: Callable = _noop_process_doc_fn,
        name: str = "spacy_op",
    ) -> None:
        super().__init__(name=name)
        self.nlp = nlp or cached_spacy_model(config.DEFAULT_SPACY_MODEL)
        self.process_doc = process_doc_fn

    def get_docs_stream(self, ds: DataStream) -> DataStream:
        """Returns DataStream of spacy Docs.
        If the data stream already contains spacy Docs then they
        are returned as-is otherwise the nlp object is used to
        create spacy Docs

        Parameters
        ----------
        ds : DataStream
            input data stream

        Returns
        ------
        out : DataStream
            A datastream containing an iterable of spacy's `Doc` objects
        """
        if ds.item_type != Doc:
            docs_with_context = self.nlp.pipe(
                zip(ds, ds.context),
                as_tuples=True,
                n_process=config.ALLOCATED_PROCESSOR_FOR_SPACY,
            )
            new_docs, context = more_itertools.unzip(docs_with_context)
            return DataStream(
                items=new_docs, applied_ops=ds.applied_ops, context=context
            )
        else:
            return ds

    def run(self, ds: DataStream) -> DataStream:
        docs_ds = self.get_docs_stream(ds)
        processed_docs = map(self.process_doc, docs_ds, docs_ds.context)
        processed_docs = (x for x in processed_docs if x is not None)
        items, context = more_itertools.unzip(processed_docs)
        return DataStream(
            items=items, applied_ops=ds.applied_ops + [self], context=context
        )
