import numpy as np

from jange import ops, stream


def test_returns_a_stream_with_doc_vectors():
    ds = stream.DataStream(["this is", "another", "sentence"])

    vector_ds = ds.apply(ops.text.doc_embedding())
    assert vector_ds.total_items == ds.total_items
    assert vector_ds.context is not None
    assert isinstance(vector_ds.items, np.ndarray)
