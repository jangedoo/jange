import itertools
import random
from typing import List

import numpy as np
import scipy.sparse as sp
import sklearn.linear_model as sklm
from sklearn.preprocessing import MultiLabelBinarizer
from spacy.util import compounding, minibatch

from jange import stream
from jange.base import TrainableMixin, accepts, produces
from jange.ops.base import ScikitBasedOperation, SpacyBasedOperation

__all__ = [
    "ClassificationResult",
    "SpacyClassificationOperation",
    "spacy_classifier",
    "SklearnClassifierOperation",
    "sgd_classifier",
    "logistic_regresssion_classifier",
    "sklearn_classifier",
]


class ClassificationResult:
    def __init__(self, label, proba, raw):
        self.label = label
        self.proba = proba
        self.raw = raw

    def __repr__(self):
        return f"ClassificationResult(label={self.label}, proba={self.proba})"


@accepts(str, strict=True)
@produces(ClassificationResult)
class SpacyClassificationOperation(SpacyBasedOperation, TrainableMixin):
    """Operation for training spacy's model for classification
    """

    def __init__(
        self,
        nlp=None,
        exclusive_classes=True,
        architecture: str = "ensemble",
        name: str = "spacy_classifier",
    ):
        super().__init__(nlp=nlp, name=name)
        self.exclusive_classes = exclusive_classes
        self.architecture = architecture

    def _get_text_cat_pipeline(self):
        if "textcat" not in self.nlp.pipe_names:
            textcat = self.nlp.create_pipe(
                "textcat",
                config={
                    "exclusive_classes": self.exclusive_classes,
                    "architecture": self.architecture,
                },
            )
            self.nlp.add_pipe(textcat, last=True)
        else:
            textcat = self.nlp.get_pipe("textcat")

        return textcat

    def _prepare_labels_for_training(self, labels: list, unique_classes):
        """Prepares labels in spacy's format for training"""
        output = []
        for label in labels:
            cats = {lbl: False for lbl in unique_classes}
            if self.exclusive_classes:
                cats[label] = True
            else:
                for lbl in label:
                    cats[lbl] = True
            output.append({"cats": cats})

        return output

    def _fit(self, ds, fit_params: dict = {}):
        if any(x not in list(fit_params.keys()) for x in ["classes", "y"]):
            raise ValueError(
                "SpacyClassificationOperation needs 'classes' and 'y' as fit_params"
                " where 'y' is an array of labels for each item and 'classes' is the"
                " distinct classes in your dataset"
            )
        n_iter = fit_params.get("n_iter", 20)
        batch_size = fit_params.get("batch_size", 128)
        y = fit_params["y"]
        unique_classes = fit_params["classes"]
        # add unique labels for the textclassifier's "memory"
        textcat = self._get_text_cat_pipeline()
        for lbl in unique_classes:
            textcat.add_label(lbl)

        # prepare label for training
        labels = self._prepare_labels_for_training(y, unique_classes)

        pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [
            pipe for pipe in self.nlp.pipe_names if pipe not in pipe_exceptions
        ]

        train_data = zip(ds, labels)

        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()
            batch_sizes = compounding(batch_size // 2, batch_size, 1.05)

            for i in range(n_iter):
                # because each iteration consumes entire training data which could exhaust
                # the generator of the DataStream we make a copy of it
                train_data, copy_train_data = itertools.tee(train_data, 2)
                losses = {}
                batches = minibatch(copy_train_data, size=batch_sizes)
                for batch in batches:
                    random.shuffle(batch)
                    texts, annotations = zip(*batch)
                    self.nlp.update(
                        texts, annotations, sgd=optimizer, drop=0.2, losses=losses
                    )
                self.logger.info(f"iteration {i} losses = {losses}")

    def _parse_prediction_result(self, pred) -> List[ClassificationResult]:
        """Parses spacy's prediction and returns a list of ClassificationResult"""
        raw = pred
        label, proba = sorted(pred.items(), key=lambda kv: kv[1], reverse=True)[0]
        return ClassificationResult(label=label, proba=proba, raw=raw)

    def _predict(self, ds: stream.DataStream) -> List[ClassificationResult]:
        textcat = self._get_text_cat_pipeline()
        preds = textcat.pipe(self.nlp.make_doc(d) for d in ds)

        return map(self._parse_prediction_result, (p.cats for p in preds))

    def run(self, ds: stream.DataStream, fit_params: dict = {}) -> stream.DataStream:
        if self.should_train:
            if ds.is_countable:
                train_ds = ds
                pred_ds = ds
            else:
                train_items, pred_items = itertools.tee(ds, 2)
                train_context, pred_context = itertools.tee(ds.context, 2)
                train_ds = stream.DataStream(train_items, context=train_context)
                pred_ds = stream.DataStream(pred_items, context=pred_context)
            self._fit(train_ds, fit_params)
        else:
            pred_ds = ds

        predictions = self._predict(pred_ds)
        return stream.DataStream(
            items=predictions, applied_ops=ds.applied_ops + [self], context=ds.context
        )


def spacy_classifier(nlp=None, name="spacy_classifier"):
    return SpacyClassificationOperation(nlp=nlp, name=name)


@accepts(np.ndarray, sp.spmatrix, strict=True)
@produces(ClassificationResult)
class SklearnClassifierOperation(ScikitBasedOperation):
    def __init__(
        self, model, exclusive_classes=True, name: str = "sklearn_classifier"
    ) -> None:
        super().__init__(model=model, predict_fn_name="predict", name=name)
        self.exclusive_classes = exclusive_classes

    def _fit(self, ds: stream.DataStream, fit_params: dict):
        required_params = ["y"]
        if self.supports_batch_training or not self.exclusive_classes:
            required_params.append("classes")
        if any(x not in list(fit_params.keys()) for x in required_params):
            raise ValueError(
                "SklearnClassifierOperation needs 'classes' and 'y' as fit_params"
                " where 'y' is an array of labels for each item and 'classes' is the"
                " distinct classes in your dataset"
            )

        if not self.exclusive_classes:
            self.label_encoder = MultiLabelBinarizer()
            self.label_encoder.fit([fit_params["classes"]])
            transformed_classes = self.label_encoder.transform(fit_params["y"])
            fit_params["y"] = transformed_classes

            if not self.supports_batch_training:
                del fit_params["classes"]

        return super()._fit(ds, fit_params)

    def _predict(self, ds, predict_params: dict = {}):
        for batch, context in self._get_batch(self.bs, ds, ds.context):
            if self.exclusive_classes:
                yield self._predict_multiclass(batch, context)
            else:
                yield self._predict_multilabel(batch, context)

    def _predict_multilabel(self, batch, context):
        results = []
        labels = self.label_encoder.inverse_transform(self.model.predict(batch))
        for label in labels:
            results.append(ClassificationResult(label=label, proba=None, raw=label))

        return results, context

    def _predict_multiclass(self, batch, context):
        results = []
        if hasattr(self.model, "predict_proba"):
            raw_output = self.model.predict_proba(batch)
            label_indices = raw_output.argmax(axis=1)
            labels, probas, raws = [], [], []
            for row_number, lbl_idx in enumerate(label_indices):
                label, proba, raw = self._parse_probabilities(
                    raw_output, row_number, lbl_idx
                )

                results.append(ClassificationResult(label=label, proba=proba, raw=raw))
        else:
            labels = self.model.predict(batch)
            probas = [1.0] * len(labels)
            raws = dict(zip(labels, probas))

            for label, proba, raw in zip(labels, probas, raws):
                results.append(ClassificationResult(label=label, proba=proba, raw=raw))

        return results, context

    def _parse_probabilities(self, raw_output, row_number, lbl_idx):
        probabilities = raw_output[row_number]

        label = self.model.classes_[lbl_idx]
        proba = probabilities[lbl_idx]
        # probability distribution
        raw = dict(zip(self.model.classes_, probabilities))
        return label, proba, raw


def sgd_classifier(exclusive_classes=True, name="sgd_classifier"):
    return SklearnClassifierOperation(
        model=sklm.SGDClassifier(loss="modified_huber"),
        exclusive_classes=exclusive_classes,
        name=name,
    )


def logistic_regresssion_classifier(
    exclusive_classes=True, name="logistic_regresssion_classifier"
):
    return SklearnClassifierOperation(
        model=sklm.LogisticRegression(), exclusive_classes=exclusive_classes, name=name
    )


def sklearn_classifier(model, exclusive_classes=True, name="sklearn_classifier"):
    return SklearnClassifierOperation(
        model=model, exclusive_classes=exclusive_classes, name=name
    )
