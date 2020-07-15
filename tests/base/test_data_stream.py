from unittest.mock import MagicMock
import pytest
from jange.base import DataStream, OperationCollection, Operation


def test_can_iterate_all_underlying_data():
    data = [1, 2, 3, 4]
    ds = DataStream(items=data)
    assert list(ds) == data


def test_all_operations_are_applied():
    data = [1, 2, 3, 4]
    ds = DataStream(applied_ops=None, items=data)

    op1 = MagicMock()
    op2 = MagicMock()

    ds.apply(op1, op2)

    op1.run.assert_called_once()
    op2.run.assert_called_once()


@pytest.mark.parametrize("items,dtype", [([1, 2, 3], int), (["a", "b", "c"], str)])
def test_item_type_of_stream(items, dtype):
    ds = DataStream(items)
    assert ds.item_type == dtype
    assert len(list(ds.items)) == len(items)


@pytest.mark.parametrize(
    "ops",
    [
        ([Operation(name="op1"), Operation(name="op2")]),
        (OperationCollection([Operation(name="op1"), Operation(name="op2")])),
    ],
)
def test_converts_applied_ops_to_operation_collection(ops):
    """Test to make sure that the applied_ops property of
    a stream is maintained as OperationCollection even if
    a list of operations is passed
    """
    ds = DataStream(items=[1], applied_ops=ops)

    assert isinstance(ds.applied_ops, OperationCollection)


@pytest.mark.parametrize("items", [[], None, tuple(), set()])
def test_raises_exception_when_items_is_finite_but_empty_or_none(items):
    with pytest.raises(ValueError):
        DataStream(items=items)


def test_checking_item_type_for_generators_does_not_consume():
    num_elements = 10
    gen = (i * 2 for i in range(num_elements))
    ds = DataStream(items=gen)
    assert ds.is_countable is False
    assert ds.item_type == int
    assert len(list(ds)) == num_elements
