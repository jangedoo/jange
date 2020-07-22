"""This module contains operations to manipulate streams
"""
import more_itertools
from jange.base import Operation, DataStream
import itertools


class FlattenOperation(Operation):
    """Operation to flatten a DataStream where each item is of type List[Any]
    into a DataStream of [Any].

    Attributes
    ----------
    distribute_context : bool
        if true, then the same context will be copied for all sub-items otherwise each
        sub-item will have a new context as <ctx_i> where ctx is the contexe associated
        with the item and i this the index of the sub-item in the item

    name : str
        name of this operation

    Example
    -------
    >>> ds = stream.DataStream(items=[[1, 2], [3, 4, 5]], context=["a", "b"])
    >>> ds.apply(flatten(distribute_context=True))
    >>> print(list(ds), list(ds.context))
    >>> [(1, 'a'), (2, 'a'), (3, 'b'), (4, 'b'), (5, 'b')]
    """

    def __init__(self, distribute_context: bool = True, name: str = "flatten") -> None:
        super().__init__(name=name)
        self.distribute_context = distribute_context

    def _flatten(self, item, ctx):
        for i, obj in enumerate(item):
            new_context = ctx if self.distribute_context else f"{ctx}_{i}"
            yield obj, new_context

    def run(self, ds: DataStream) -> DataStream:
        flat = itertools.chain.from_iterable(map(self._flatten, ds, ds.context))

        items, context = more_itertools.unzip(flat)
        return DataStream(
            items=items, applied_ops=ds.applied_ops + [self], context=context
        )


def flatten(distribute_context: bool = True, name: str = "flatten"):
    """Returns an operation to flatten a DataStream where each item is of type List[Any]
    into a DataStream of [Any].

    Attributes
    ----------
    distribute_context : bool
        if true, then the same context will be copied for all sub-items otherwise each
        sub-item will have a new context as <ctx_i> where ctx is the contexe associated
        with the item and i this the index of the sub-item in the item

    name : str
        name of this operation

    Returns
    -------
    FlattenOperation

    """
    return FlattenOperation(distribute_context=distribute_context, name=name)
