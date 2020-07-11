from unittest.mock import MagicMock
import pytest
from jange.stream import DataStream


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
