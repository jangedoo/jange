from typing import Any, List, Optional, Iterable
import cytoolz


class DataStream:
    """A class representing a stream of data. A data stream is created as
    a result of some operation. DataStream object can be iterated which
    basically iterates through the underlying data. The underlying data
    is stored in `items` attribute which can be any iterable object.
    
    Parameters
    ----------
    items : iterable
        an iterable that contains the raw data

    applied_ops : Optional[List[Operation]]
        a list of operations that were applied to create this stream of data    

    Example
    -------
    >>> ds = DataStream(items=[1, 2, 3])
    >>> print(list(ds))
    >>> [1, 2, 3]


    Attributes
    ----------
    applied_ops : List[Operation]
        a list of operations that were applied to create this stream of data

    items : iterable
        an iterable that contains the raw data
    """

    def __init__(
        self,
        items: Iterable[Any],
        applied_ops: Optional[List] = None,
        context: Optional[Iterable[Any]] = None,
    ):
        if context is not None:
            self.items = items
            self.context = context
        else:
            self.context, self.items = zip(*enumerate(items))

        self.applied_ops = applied_ops or []

    def __iter__(self):
        for item in self.items:
            yield item

    @property
    def item_type(self):
        first, items = cytoolz.peek(self.items)
        self.items = items
        return type(first)

    def apply(self, *ops):
        x = self
        for op in ops:
            x = op.run(x)
        return x


class Operation:
    def run(self, ds: DataStream) -> DataStream:
        raise NotImplementedError()

    def __call__(self, ds: DataStream) -> DataStream:
        return self.run(ds)


class TrainableMixin:
    def __init__(self) -> None:
        self.should_train = True
