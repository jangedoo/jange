import itertools
import random
from typing import List

from spacy.util import compounding, minibatch

from jange.ops.base import SpacyModelPicklerMixin
from .result import ClassificationResult


class ClassificationModel:
    def fit(self, x, y):
        raise NotImplementedError()

    def predict(self, x) -> List[ClassificationResult]:
        raise NotImplementedError()


class SpacyClassificationModel(ClassificationModel, SpacyModelPicklerMixin):
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
        train_data = zip(x, labels)

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
