import jange
from jange import ops
from jange import stream


def test_disable_training_fn():
    tfidf1 = ops.text.tfidf()
    tfidf2 = ops.text.tfidf()
    tfidf2.should_train = False

    operations = [ops.text.lowercase(), tfidf1, tfidf2]

    with ops.utils.disable_training(operations) as training_disabled_ops:
        assert len(operations) == len(training_disabled_ops)
        # check that all trainable operations have should_train set to False
        assert all(
            op.should_train is False
            for op in operations
            if isinstance(op, jange.base.TrainableMixin)
        )

    # check that trainable operations have their `should_train` value set to
    # what was originally there before
    assert tfidf1.should_train == True
    assert tfidf2.should_train == False
