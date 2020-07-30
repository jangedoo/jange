import functools
import warnings
from typing import Any, Iterable, List, Optional

import cytoolz
import more_itertools
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
            try:
                return next((op for op in self if op.name == name))
            except StopIteration:
                raise LookupError(f"Operation with name {name} was not found")
        else:
            return OperationCollection(op for op in self if op.name == name)

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
            raise LookupError(f"Operation with name {name} was not found")


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
        if items is None:
            raise ValueError("items cannot be None")

        # items is countable
        # if items count is 0 then raise Exception
        # if context is None then generate a list of context
        # if context is generator then raise Exception
        # if context is countable and does not have same length as items
        # then raise Exception

        if self._is_countable(items):
            if self._count_items(items) == 0:
                raise ValueError(
                    f"items must contain atleast one element but got {items}"
                )
            elif context is None:
                context = list(range(len(items)))
            elif not self._is_countable(context):
                raise ValueError(
                    f"context cannot be a generator when items is not a generator but got {context}"
                )
            else:
                items_len = self._count_items(items)
                context_len = self._count_items(context)
                if items_len != context_len:
                    raise ValueError(
                        "items and context should have same length "
                        f"but received items with length={items_len} and context with length={context_len}"
                    )

        # items is not countable i.e. a generator
        # if context is None then create a context
        else:
            if context is None:
                context, items = more_itertools.unzip(enumerate(items))

        self.items = items
        self.context = context

        if applied_ops and not isinstance(applied_ops, OperationCollection):
            applied_ops = OperationCollection(applied_ops)
        self.applied_ops = applied_ops or OperationCollection()

    def _is_countable(self, x):
        return hasattr(x, "__len__")

    def _count_items(self, items):
        if not self._is_countable(items):
            raise AttributeError(
                "Length of this datastream cannot be determined because the items are from a generator"
            )

        if issparse(items):
            return items.shape[0]
        else:
            return len(items)

    def __iter__(self):
        for item in self.items:
            yield item

    @property
    def total_items(self):
        return self._count_items(self.items)

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

    def apply(self, *ops, result_collector: dict = None, op_kwargs: dict = {}):
        x = self
        for op in ops:
            kwargs = op_kwargs.get(op.name, {})
            x = op.run(x, **kwargs)
            if result_collector is not None:
                result_collector[op] = x
        return x

    def __repr__(self) -> str:
        item_type = self.item_type
        total_items = self.total_items if self.is_countable else "unknown"
        return f"DataStream(item_type={item_type}, is_finite={self.is_countable}, total_items={total_items})"


class Operation:
    accepts_types = []
    produces_types = []

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__()
        name = name or self.__class__.__name__
        self.name = name

    def run(self, ds: DataStream) -> DataStream:
        raise NotImplementedError()

    def __call__(self, ds: DataStream) -> DataStream:
        return self.run(ds)


class FunctionOperation(Operation):
    def __init__(self, name: str, fn):
        super().__init__(name=name if name else fn.__name__)
        self.fn = fn

    def run(self, ds):
        return self.fn(ds)


class TrainableMixin:
    def __init__(self) -> None:
        self.should_train = True


def accepts(*types, strict=False):
    """Class decorator for specifying what types of items in the DataStream are supported by the Operation class
    """

    def _accepts(cls: Operation):
        cls.strict_validation = strict
        cls.accepts_types = types
        f = getattr(cls, "run")

        @functools.wraps(f)
        def wrapper(self, ds: DataStream, *args, **kwargs):
            item_type = ds.item_type
            if self.accepts_types and item_type not in self.accepts_types:
                msg = f"Operation {self} expected DataStream having item_type = any{self.accepts_types} but got {item_type}"
                if self.strict_validation:
                    raise Exception(msg)
                else:
                    warnings.warn(msg)
            return f(self, ds, *args, **kwargs)

        cls.run = wrapper
        return cls

    return _accepts


def produces(*types):
    """Class decorator for specifying what type of item will be in the DataStream that this Operation returns
    """

    def _produces(cls):
        cls.produces_types = types
        return cls

    return _produces


def operation(name=None, accepts=(), produces=(), strict=False):
    """Decorator for returning FunctionOperation from the given function

    Example
    -------
    >>> @operation(accepts=str, produces=str)
    >>> def uppercase_op(ds):
    >>>     return (text.upper() for text in ds)
    >>> op = uppercase_op()
    >>> op.run(stream.DataStream(["hi", "hello"]))
    """

    def _operation(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            o = FunctionOperation(name=name if name else f.__name__, fn=f)
            o.strict_validation = strict
            o.accepts_types = (
                accepts if isinstance(accepts, (list, tuple)) else [accepts]
            )
            o.produces_types = (
                produces if isinstance(produces, (list, tuple)) else [produces]
            )
            return o

        return wrapper

    return _operation
