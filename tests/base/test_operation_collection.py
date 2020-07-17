import pytest
from jange.base import OperationCollection, Operation


def test_operation_collection_find_by_name():
    op1, op2 = Operation(name="op1"), Operation(name="op2")
    col = OperationCollection([op1, op2])
    assert op1 == col.find_by_name("op1")
    assert op2 == col.find_by_name("op2")

    assert [op1] == col.find_by_name("op1", first_only=False)

    with pytest.raises(LookupError):
        col.find_by_name("non_existing_name")


def test_find_up_to():
    op1, op2, op3, op4 = [Operation(name=f"{i}") for i in range(4)]
    col = OperationCollection([op1, op2, op3, op4])

    assert col.find_up_to("0") == OperationCollection([op1])
    assert col.find_up_to("1") == OperationCollection([op1, op2])
    assert col.find_up_to("2") == OperationCollection([op1, op2, op3])
    assert col.find_up_to("3") == OperationCollection([op1, op2, op3, op4])

    with pytest.raises(LookupError):
        assert col.find_up_to("non_existing_name") == OperationCollection()


def test_operation_collection_concat_two_collections():
    op1, op2, op3, op4 = [Operation(name=f"{i}") for i in range(4)]
    col1 = OperationCollection([op1, op2])
    col2 = OperationCollection([op3, op4])

    expected = OperationCollection([op1, op2, op3, op4])
    actual = col1 + col2
    assert actual == expected
