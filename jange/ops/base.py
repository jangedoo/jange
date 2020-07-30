import itertools
from typing import Callable, Optional

import cytoolz
import more_itertools
import numpy as np
import scipy.sparse as sparse
from spacy.language import Language
from spacy.tokens import Doc

from jange import config
from jange.base import Operation, TrainableMixin, accepts, produces
from jange.stream import DataStream

from .utils import cached_spacy_model


class SpacyModelPicklerMixin:
    """Class intented to be inherited by classes that use
    spacy's model so that the spacy's model is not pickled.
    Instead only the path to the mode is pickled
    """

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


@accepts(str, Doc)
@produces(str, Doc)
class SpacyBasedOperation(Operation, SpacyModelPicklerMixin):
    """Base class for operations using spacy's langauge model

    Parameters
    ----------
    nlp : Optional[Language]
        spacy's language model. if None, then model defined in config.DEFAULT_SPACY_MODEL is used

    process_doc_fn : Callable
        a function that accepts a document and context and returns a tuple <object, context>.
        Default function is an identity function. This function is called for each document in
        the stream

    name : str
        name of this operation
    """

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


class ScikitBasedOperation(Operation, TrainableMixin):
    """Base class for operations using scikit-learn's Estimators

    Attributes
    ----------
    model : any sklearn Estimator

    predict_fn_name : str
        name of function or attribute in the model to get predictions.
        Usually this is transform, predict or kneighbors. For models
        that do not support predicting on new dataset, this should be
        the name of attribute that holds the data. E.g. for clustering
        models like DBSCAN, AgglomerativeClustering it would be `labels_`
        or for dimension reduction approaches like TSNE, SpectralEmbedding
        it would be `embedding_`

    Example
    -------
    >>> import sklearn.linear_model as sklm
    >>> import sklearn.decomposition as skdecomp
    >>> import sklearn.cluster as skcluster
    >>> op1 = ScikitBasedOperation(sklm.SGDClassifier(), predict_fn_name="predict")
    >>> op2 = ScikitBasedOperation(skdecomp.PCA(15), predict_fn_name="transform")
    >>> op3 = ScikitBasedOperation(skcluster.DBSCAN(), predict_fn_name="labels_")
    """

    def __init__(
        self,
        model,
        predict_fn_name: str,
        batch_size: int = 1000,
        name: str = "sklearn_op",
    ):
        super().__init__(name=name)
        self.model = model
        self.predict_fn_name = predict_fn_name
        self.bs = batch_size

    @property
    def supports_batch_training(self):
        return hasattr(self.model, "partial_fit")

    @property
    def can_predict_on_new(self):
        """Returns whether sklearn's estimator can predict on unseen data

        It checks whether the given `predict_fn_name` is present on the model
        and if it exists then checks whether it is a function or not.

        Note
        ----
        Estimators not supporting unseen data prediction will populate some attribute
        like `labels_` or `embeddings_` only after the model has been trained..

        Returns
        -------
        bool
            If the estimator can predict on new dataset
        """
        has_attrib = hasattr(self.model, self.predict_fn_name)
        if has_attrib:
            attrib = getattr(self.model, self.predict_fn_name)
            # if this is a function then it predicts on new input
            return callable(attrib)
        return False

    def _get_batch(self, bs: int, x, y=None):
        y_x_pairs = zip(y, x) if y else enumerate(x)
        for batch in cytoolz.partition_all(bs, y_x_pairs):
            batch_y, batch_x = more_itertools.unzip(batch)
            X, Y = list(batch_x), list(batch_y)
            if sparse.issparse(X[0]):
                X = sparse.vstack(X)
            elif isinstance(X[0], np.ndarray):
                X = np.vstack(X)
            yield X, Y

    def _train(self, ds: DataStream, fit_params: dict = {}):
        if self.supports_batch_training:
            bs = self.bs
            items = ds.items
        else:
            items = list(ds)
            bs = len(items)

        labels = fit_params.pop("labels", None)
        for x, y in self._get_batch(bs, items, labels):
            if self.supports_batch_training:
                self.model.partial_fit(x, y, **fit_params)
            else:
                self.model.fit(x, y, **fit_params)

    def _predict(self, ds, predict_params: dict = {}):
        # if this cannot predict on new then return the value
        # stored in some attribute
        if not self.can_predict_on_new:
            preds = getattr(self.model, self.predict_fn_name)
            yield preds, ds.context
        else:
            predict_fn = getattr(self.model, self.predict_fn_name)
            for batch, context in self._get_batch(self.bs, ds, ds.context):
                preds = predict_fn(batch, **predict_params)
                yield preds, context

    def run(self, ds, fit_params: dict = {}, predict_params: dict = {}):
        if not self.can_predict_on_new:
            self.should_train = True

        if self.should_train:
            if ds.is_countable:
                train_ds = ds
                pred_ds = ds
            else:
                train_items, pred_items = itertools.tee(ds, 2)
                train_context, pred_context = itertools.tee(ds.context, 2)
                train_ds = DataStream(train_items, context=train_context)
                pred_ds = DataStream(pred_items, context=pred_context)

            self._train(train_ds, fit_params)
        else:
            pred_ds = ds
        preds, context = more_itertools.unzip(self._predict(pred_ds, predict_params))
        preds = itertools.chain.from_iterable(preds)
        context = itertools.chain.from_iterable(context)
        return DataStream(
            items=preds, context=context, applied_ops=ds.applied_ops + [self]
        )
