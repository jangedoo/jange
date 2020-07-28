import numpy as np

from jange import ops, stream


def test_returns_a_stream_with_doc_vectors():
    ds = stream.DataStream(["this is", "another", "sentence"])

    vector_ds = ds.apply(ops.text.embedding.doc_embedding())
    vectors = list(vector_ds)
    context = list(vector_ds.context)
    assert len(vectors) == ds.total_items
    assert len(context) == len(vectors)
    assert all(isinstance(v, np.ndarray) for v in vectors)
