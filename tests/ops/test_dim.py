from unittest.mock import MagicMock

import pytest

from jange.ops import dim, stream


def test_pca_returns_proper_operation():
    n_dim = 3
    op = dim.pca(n_dim=n_dim)
    assert isinstance(op, dim.DimensionReductionOperation)
    assert isinstance(op.model, dim.skd.PCA)
    assert op.model.n_components == n_dim


def test_tsne_returns_proper_operation():
    n_dim = 3
    op = dim.tsne(n_dim=n_dim)
    assert isinstance(op, dim.DimensionReductionOperation)
    assert isinstance(op.model, dim.skm.TSNE)
    assert op.model.n_components == n_dim


def test_raises_error_if_invalid_model_passed():
    with pytest.raises(ValueError):
        dim.DimensionReductionOperation(model=MagicMock())


@pytest.mark.parametrize("model_class", dim.SUPPORTED_CLASSES)
@pytest.mark.parametrize("should_train", [True, False])
def test_calls_appropriate_underlying_methods_for_training_and_prediction(
    model_class, should_train
):
    model = MagicMock(spec=model_class)
    context = ["context1", "context2"]
    ds = stream.DataStream([[0, 1, 2], [3, 4, 5]], context=context)

    op = dim.DimensionReductionOperation(model=model)
    op.should_train = should_train

    assert op.should_train == should_train

    test_return_value = [[0, 1], [3, 4]]
    if model_class in dim.ALGORITHMS_NOT_SUPPORTING_NEW_INFERENCE:
        model.embedding_ = test_return_value
        output = op.run(ds)
        # assert that fit_transform was called regardless of state of `should_train`
        model.fit.assert_called_once()
        assert list(output.items) == test_return_value
        assert list(output.context) == context

    elif model_class in dim.ALGORITHMS_SUPPORTING_NEW_INFERENCE:
        model.transform.return_value = test_return_value
        output = op.run(ds)
        if should_train:
            model.fit.assert_called_once()
        else:
            model.fit.assert_not_called()
            model.transform.assert_called_once()

        assert list(output.items) == test_return_value
        assert list(output.context) == context
    else:
        raise Exception("model class not expected")
