from typing import Any, List, Optional, Iterable
import cytoolz
from scipy.sparse import issparse


class OperationCollection(list):
    def __add__(self, other):
        result = super().__add__(other)
        return OperationCollection(result)

    def find_by_name(self, name: str, first_only=True):
        """Finds the operation(s) by its name.

        Parameters
        ----------
        name : str
            name of the operation
        first_only : bool, optional
            if True then returns first operation with
            the given name otherwise returns all operations
            matching the given name, by default True

        Returns
        -------
        Operation

        Example
        -------
        >>> ds.applied_ops.find_by_name(name="tfidf")
        >>> ds.applied_ops.find_by_name(name="token_filter", first_only=False)
        """
        if first_only:
            return next((op for op in self if op.name == name))
        else:
            return [op for op in self if op.name == name]

    def find_up_to(self, name: str):
        """Returns all operations upto the operation matching the given name

        Parameters
        ----------
        name : str
            name of the operation

        Returns
        -------
        OperationCollection[Operation]
            Collection of operations that were applied upto the operation with
            given name

        Example
        -------
        To return all operations applied to a DataStream up to an operation with
        name 'tfidf'
        >>> feature_extraction_ops = ds.find_up_to(name="tfidf")
        """
        found = False
        output = OperationCollection()
        for op in self:
            output.append(op)
            if op.name == name:
                found = True
                break

        if found:
            return output
        else:
            return []


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
        self._validate_items_or_raise(items)

        self.items = items
        self.context = context

        if applied_ops and not isinstance(applied_ops, OperationCollection):
            applied_ops = OperationCollection(applied_ops)
        self.applied_ops = applied_ops or OperationCollection()

    def _is_countable(self, x):
        return hasattr(x, "__len__")

    def _validate_items_or_raise(self, items):
        if items is None:
            raise ValueError("items cannot be None")

        if self._is_countable(items):
            # sparse matrix don't support calling len
            # so use getnnz()
            if issparse(items):
                count = items.getnnz()
            else:
                count = len(items)

            if count == 0:
                raise ValueError(
                    f"items must have atleast one element. received {items}"
                )

    def __iter__(self):
        for item in self.items:
            yield item

    @property
    def total_items(self):
        if not self.is_countable:
            raise AttributeError(
                "Length of this datastream cannot be determined because the items are from a generator"
            )

        if issparse(self.items):
            return self.items.shape[0]
        else:
            return len(self.items)

    @property
    def is_countable(self) -> bool:
        return self._is_countable(self.items)

    @property
    def item_type(self):
        if self.is_countable:
            return type(self.items[0])
        else:
            first, items = cytoolz.peek(self.items)
            self.items = items
            return type(first)

    def apply(self, *ops):
        x = self
        for op in ops:
            x = op.run(x)
        return x

    def __repr__(self) -> str:
        item_type = self.item_type
        total_items = self.total_items if self.is_countable else "unknown"
        return f"DataStream(item_type={item_type}, is_finite={self.is_countable}, total_items={total_items})"


class Operation:
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__()
        name = name or self.__class__.__name__
        self.name = name

    def run(self, ds: DataStream) -> DataStream:
        raise NotImplementedError()

    def __call__(self, ds: DataStream) -> DataStream:
        return self.run(ds)


class TrainableMixin:
    def __init__(self) -> None:
        self.should_train = True
