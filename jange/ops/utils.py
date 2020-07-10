from contextlib import contextmanager
from typing import List, Optional
from .base import Operation, TrainableMixin


@contextmanager
def disable_training(ops: List[Operation]) -> List[Operation]:
    """Disables the "training mode" of operations so that these
    operations can be used for inference.

    Some operations like tfidf, kmeans, sgd etc. need to learn
    from the data and by default they are in "training mode"
    which will learn from the stream that is passed to them. Once
    these have been trained we want to use it in production so
    we need to disable the training.

    Example
    -------
    >>> with ops.utils.disable_training(stream.applied_ops) as new_ops:
    >>>     ops.utils.save(new_ops, path="./operations")

    Parameters
    ----------
    ops : List[Operation]
        a list of operations
    """
    original_mode = {}
    try:
        for i, op in enumerate(ops):
            if isinstance(op, TrainableMixin):
                original_mode[i] = op.should_train
                op.should_train = False

        yield ops
    finally:
        for i, op in enumerate(ops):
            if isinstance(op, TrainableMixin):
                op.should_train = original_mode[i]


def save(ops: List[Operation], path: str, replace: bool = False):
    pass


def load(path: str) -> List[Operation]:
    pass
