from jange import ops, stream


def test_token_extraction():
    ds = stream.DataStream(items=["this is a sentence", "woo hoo", ""])
    out_ds = ds.apply(ops.text.extract.tokens())

    expected = [["this", "is", "a", "sentence"], ["woo", "hoo"], []]
    assert list(out_ds.items) == expected


def test_sentence_extraction():
    texts = ["this is a text. with two sentences.", "this is with single sentence"]
    out_ds = stream.DataStream(items=texts).apply(ops.text.extract.sentences())
    expected_items = [
        ["this is a text.", "with two sentences."],
        ["this is with single sentence"],
    ]
    expected_context = [0, 1]

    actual_items, actual_context = list(out_ds.items), list(out_ds.context)
    assert actual_items == expected_items
    assert actual_context == expected_context
