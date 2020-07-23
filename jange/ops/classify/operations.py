import sklearn.linear_model as sklm

from jange.base import DataStream, Operation, TrainableMixin

from .models import (
    ClassificationModel,
    ScikitClassificationModel,
    SpacyClassificationModel,
)


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


def sgd_classifier(labels: list, name="sgd_classifier"):
    model = ScikitClassificationModel(sklm.SGDClassifier(loss="modified_huber"))
    return ClassificationOperation(model=model, labels=labels, name=name)


def logistic_regresssion_classifier(
    labels: list, name="logistic_regresssion_classifier"
):
    model = ScikitClassificationModel(sklm.LogisticRegression())
    return ClassificationOperation(model=model, labels=labels, name=name)


def sklearn_classifier(model, labels: list, name="sklearn_classifier"):
    return ClassificationOperation(model=model, labels=labels, name=name)
