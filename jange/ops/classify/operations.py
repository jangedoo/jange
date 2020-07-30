import sklearn.linear_model as sklm

from jange.base import DataStream, Operation, TrainableMixin
from jange.ops.base import ScikitBasedOperation

from .models import (
    ClassificationModel,
    SpacyClassificationModel,
)

__all__ = [
    "ClassificationResult",
    "SklearnClassifierOperation",
    "sgd_classifier",
    "logistic_regresssion_classifier",
    "sklearn_classifier",
]


class ClassificationOperation(Operation, TrainableMixin):
    """Operation to classify inputs into classes or labels.
    """

    def __init__(
        self, model: ClassificationModel, labels: None, name="classification"
    ) -> None:
        super().__init__(name=name)
        self.model = model
        self._labels = labels

    def run(self, ds: DataStream) -> DataStream:
        if self.should_train:
            if self._labels:
                self.model.fit(ds.items, self._labels)
            else:
                raise ValueError(
                    "labels must be provided when should_train property is True"
                )
            self.model.fit(ds.items, self._labels)

        preds = self.model.predict(ds)
        return DataStream(
            items=preds, applied_ops=ds.applied_ops + [self], context=ds.context
        )


def spacy_classifier(nlp, labels: list, name="spacy_classifier"):
    model = SpacyClassificationModel(nlp=nlp)
    return ClassificationOperation(model=model, labels=labels, name=name)


class ClassificationResult:
    def __init__(self, label, proba, raw):
        self.label = label
        self.proba = proba
        self.raw = raw

    def __repr__(self):
        return f"ClassificationResult(label={self.label}, proba={self.proba})"


class SklearnClassifierOperation(ScikitBasedOperation):
    def __init__(self, model, name: str = "scikit_classifier") -> None:
        super().__init__(model=model, predict_fn_name="predict", name=name)

    def _predict(self, ds, predict_params: dict = {}):
        for batch, context in self._get_batch(self.bs, ds, ds.context):
            results = []
            if hasattr(self.model, "predict_proba"):
                raw_output = self.model.predict_proba(batch)
                label_indices = raw_output.argmax(axis=1)
                labels, probas, raws = [], [], []
                for row_number, lbl_idx in enumerate(label_indices):
                    label, proba, raw = self._parse_probabilities(
                        raw_output, row_number, lbl_idx
                    )

                    results.append(
                        ClassificationResult(label=label, proba=proba, raw=raw)
                    )
            else:
                labels = self.model.predict(batch)
                probas = [1.0] * len(labels)
                raws = dict(zip(labels, probas))

                for label, proba, raw in zip(labels, probas, raws):
                    results.append(
                        ClassificationResult(label=label, proba=proba, raw=raw)
                    )

            yield results, context

    def _parse_probabilities(self, raw_output, row_number, lbl_idx):
        probabilities = raw_output[row_number]

        label = self.model.classes_[lbl_idx]
        proba = probabilities[lbl_idx]
        # probability distribution
        raw = dict(zip(self.model.classes_, probabilities))
        return label, proba, raw


def sgd_classifier(name="sgd_classifier"):
    return SklearnClassifierOperation(
        model=sklm.SGDClassifier(loss="modified_huber"), name=name
    )


def logistic_regresssion_classifier(name="logistic_regresssion_classifier"):
    return SklearnClassifierOperation(model=sklm.LogisticRegression(), name=name)


def sklearn_classifier(model, name="sklearn_classifier"):
    return SklearnClassifierOperation(model=model, name=name)
