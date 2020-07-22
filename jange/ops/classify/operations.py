from jange.base import DataStream, Operation, TrainableMixin

from .models import ClassificationModel, SpacyClassificationModel
from .result import ClassificationResult


class ClassificationOperation(Operation, TrainableMixin):
    def __init__(
        self, model: ClassificationModel, labels: None, name="classification"
    ) -> None:
        super().__init__(name=name)
        self.model = model
        self._labels = labels

    def run(self, ds: DataStream) -> DataStream:
        if self.should_train:
            self.model.fit(ds.items, self._labels)

        preds = self.model.predict(ds)
        return DataStream(
            items=preds, applied_ops=ds.applied_ops + [self], context=ds.context
        )


def spacy_classifier(nlp, labels, name="spacy_classifier"):
    model = SpacyClassificationModel(nlp=nlp)
    return ClassificationOperation(model=model, labels=labels, name=name)
