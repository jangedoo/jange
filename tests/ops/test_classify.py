import itertools

import numpy as np
import pytest
import sklearn.tree as sktree
from sklearn.naive_bayes import GaussianNB

from jange import ops, stream


@pytest.fixture
def multiclass_labels():
    return ["cat 1", "cat 2", "cat 3"]


@pytest.fixture
def multilabel_labels():
    return [["cat 1", "cat 2"], ["cat 2"], ["cat 1", "cat 2"]]


@pytest.fixture
def feature_vectors():
    return np.random.uniform(size=(3, 10))


@pytest.fixture
def texts():
    return ["this is sentence 1", "completely different one", "the background is black"]


@pytest.mark.parametrize(
    "classifier",
    [
        ops.classify.sgd_classifier(name="cls"),
        ops.classify.logistic_regresssion_classifier(name="cls"),
        ops.classify.sklearn_classifier(name="cls", model=GaussianNB()),
    ],
)
def test_sklearn_classifiers_work_with_multiclass_label(
    classifier, feature_vectors, multiclass_labels
):
    vec_ds = stream.DataStream(feature_vectors, context=multiclass_labels)
    fit_params = {"y": multiclass_labels}
    if classifier.supports_batch_training:
        fit_params["classes"] = list(set(multiclass_labels))

    pred_ds = vec_ds.apply(classifier, op_kwargs={"cls": {"fit_params": fit_params}},)

    preds = list(pred_ds)
    assert len(preds) == len(feature_vectors)
    assert all(isinstance(p, ops.classify.ClassificationResult) for p in preds)
    assert all(isinstance(p.label, str) for p in preds)


@pytest.mark.parametrize(
    "classifier",
    [
        ops.classify.sklearn_classifier(
            exclusive_classes=False, name="cls", model=sktree.DecisionTreeClassifier()
        ),
    ],
)
def test_sklearn_classifiers_work_with_multilabel_label(
    classifier, feature_vectors, multilabel_labels
):
    vec_ds = stream.DataStream(feature_vectors, context=multilabel_labels)
    fit_params = {"y": multilabel_labels}
    if classifier.supports_batch_training or not classifier.exclusive_classes:
        fit_params["classes"] = list(
            set(itertools.chain.from_iterable(multilabel_labels))
        )

    pred_ds = vec_ds.apply(classifier, op_kwargs={"cls": {"fit_params": fit_params}},)

    preds = list(pred_ds)
    assert len(preds) == len(feature_vectors)
    assert all(isinstance(p, ops.classify.ClassificationResult) for p in preds)
    assert all(isinstance(p.label, (list, tuple)) for p in preds)
