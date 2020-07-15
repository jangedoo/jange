from jange.base import OperationCollection, Operation


def test_operation_collection_find_by_name():
    op1, op2 = Operation(name="op1"), Operation(name="op2")
    col = OperationCollection([op1, op2])
    assert op1 == col.find_by_name("op1")
    assert op2 == col.find_by_name("op2")

    assert [op1] == col.find_by_name("op1", first_only=False)


def test_operation_collection_concat_two_collections():
    op1, op2, op3, op4 = [Operation(name=f"{i}") for i in range(4)]
    col1 = OperationCollection([op1, op2])
    col2 = OperationCollection([op3, op4])

    expected = OperationCollection([op1, op2, op3, op4])
    actual = col1 + col2
    assert actual == expected
