import itertools
import random
from typing import List

from spacy.util import compounding, minibatch

from jange.ops.base import SpacyModelPicklerMixin

from .result import ClassificationResult


class ClassificationModel:
    """Inteface for classes that wrap models from other
    libraries
    """

    def fit(self, x, y):
        """Train a model using features `x` and labels `y`
        """
        raise NotImplementedError()

    def predict(self, x) -> List[ClassificationResult]:
        """Returns predictions for the given input

        Parameters
        ----------
        x : Iterable
            an iterable of inputs

        Returns
        -------
        List[ClassificationResult]
            a list of classification result
        """
        raise NotImplementedError()


class ScikitClassificationModel:
    """Wrapper class for classification models available in scikit-learn

    Attributes
    ----------
    model : Any
        any scikit-learn model that has fit and predict method
    """

    def __init__(self, model) -> None:
        if not hasattr(model, "fit"):
            raise ValueError("model should have fit method to train it")
        if not hasattr(model, "predict"):
            raise ValueError("model should have predict function to get predictions")
        self.model = model

    def fit(self, x, y):
        self.mode.fit(x, y)

    def predict(self, x) -> List[ClassificationResult]:
        if hasattr(self.model, "predict_proba"):
            raw_output = self.model.predict_proba(x)
            label_indices = raw_output.argmax(axis=1)
            labels, probas, raws = [], [], []
            for row_number, lbl_idx in enumerate(label_indices):
                label, proba, raw = self._parse_probabilities(
                    raw_output, row_number, lbl_idx
                )

                labels.append(label)
                probas.append(proba)
                raws.append(raw)
        else:
            labels = self.model.predict(x)
            probas = [1.0] * len(labels)
            raws = [[]] * len(labels)

        output = []
        for label, proba in zip(labels, probas):
            r = ClassificationResult(label=label, proba=proba, raw=(label, proba))
            output.append(r)

        return output

    def _parse_probabilities(self, raw_output, row_number, lbl_idx):
        probabilities = raw_output[row_number]

        label = self.model.classes_[lbl_idx]
        proba = probabilities[lbl_idx]
        # probability distribution
        raw = dict(zip(self.model.classes_, probabilities))
        return label, proba, raw


class SpacyClassificationModel(ClassificationModel, SpacyModelPicklerMixin):
    """Wrapper class for training spacy's model for classification
    """

    def __init__(self, nlp):
        self.nlp = nlp

    def _get_text_cat_pipeline(self):
        if "textcat" not in self.nlp.pipe_names:
            textcat = self.nlp.create_pipe(
                "textcat",
                config={"exclusive_classes": True, "architecture": "ensemble"},
            )
            self.nlp.add_pipe(textcat, last=True)
        else:
            textcat = self.nlp.get_pipe("textcat")

        return textcat

    def _is_single_class_classification(self, labels):
        return isinstance(labels[0], (str, int, bool))

    def _get_unique_labels(self, labels):
        if self._is_single_class_classification(labels):
            unique_labels = set(labels)
        else:
            unique_labels = set(itertools.chain.from_iterable(labels))

        return unique_labels

    def _prepare_labels_for_training(self, labels: list):
        """Prepares labels in spacy's format for training"""
        is_single_class = self._is_single_class_classification(labels)

        unique_labels = self._get_unique_labels(labels)

        output = []
        for label in labels:
            cats = {lbl: False for lbl in unique_labels}
            if is_single_class:
                cats[label] = True
            else:
                for lbl in label:
                    cats[lbl] = True
            output.append({"cats": cats})

        return output

    def _parse_prediction_results(self, predictions) -> List[ClassificationResult]:
        """Parses spacy's prediction and returns a list of ClassificationResult"""
        output = []
        for pred in predictions:
            raw = pred
            label, proba = sorted(pred.items(), key=lambda kv: kv[1], reverse=True)[0]
            output.append(ClassificationResult(label=label, proba=proba, raw=raw))

        return output

    def fit(self, x, y, **kwargs):
        n_iter = kwargs.get("n_iter", 20)

        # add unique labels for the textclassifier's "memory"
        textcat = self._get_text_cat_pipeline()
        for lbl in self._get_unique_labels(y):
            textcat.add_label(lbl)

        # prepare label for training
        labels = self._prepare_labels_for_training(y)
        train_data = list(zip(x, labels))

        pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
        other_pipes = [
            pipe for pipe in self.nlp.pipe_names if pipe not in pipe_exceptions
        ]

        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()
            batch_sizes = compounding(4.0, 32.0, 1.001)
            for _ in range(n_iter):
                losses = {}
                random.shuffle(train_data)
                batches = minibatch(train_data, size=batch_sizes)
                for batch in batches:
                    texts, annotations = zip(*batch)
                    self.nlp.update(
                        texts, annotations, sgd=optimizer, drop=0.2, losses=losses
                    )

    def predict(self, x) -> List[ClassificationResult]:
        textcat = self._get_text_cat_pipeline()
        preds = list(textcat.pipe(self.nlp.make_doc(d) for d in x))
        return self._parse_prediction_results((p.cats for p in preds))
