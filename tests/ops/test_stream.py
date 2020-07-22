from jange import ops, stream


def test_flatten():
    input_ds = stream.DataStream(items=[[1, 2], [3, 4, 5]], context=["a", "b"])
    ds = input_ds.apply(ops.stream.flatten(distribute_context=True))

    assert list(ds) == [1, 2, 3, 4, 5]
    assert list(ds.context) == ["a", "a", "b", "b", "b"]

    ds = input_ds.apply(ops.stream.flatten(distribute_context=False))
    assert list(ds) == [1, 2, 3, 4, 5]
    assert list(ds.context) == ["a_0", "a_1", "b_0", "b_1", "b_2"]
