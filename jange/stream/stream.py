from typing import Any, List, Optional, Union, Iterable
import pandas as pd


class DataStream:
    """A class representing a stream of data. A data stream is created as
    a result of some operation. DataStream object can be iterated which
    basically iterates through the underlying data. The underlying data
    is stored in `items` attribute which can be any iterable object.
    
    Example
    -------
    create a stream of data from a list of numbers
    >>> ds = DataStream(items=[1, 2, 3])
    >>> print(list(ds))
    [1, 2, 3]


    Attributes
    ----------
    applied_ops : List[Operation]
        a list of operations that were applied to create this stream of data

    items : iterable
        an iterable that contains the raw data
    """

    def __init__(self, items: Iterable[Any], applied_ops: Optional[List] = None):
        self.items = items
        self.applied_ops = applied_ops or []

    def __iter__(self):
        for i in self.items:
            yield i

    def apply(self, *ops):
        x = self
        for op in ops:
            x = op.run(x)
        return x


class DataFrameStream(DataStream):
    """Represents a stream of data by iterating over the rows in a
    pandas DataFrame object.

    Example
    -------
    >>> df = pd.DataFrame([{"text": "text 1", "id": "1"}, {"text": "text 2", "id": "2"}])
    >>> ds = DataFrameStream(df=df, columns="text")
    >>> print(list(ds))
    ["text 1", "text 2"]
    >>> ds = DataFrameStream(df=df, columns=["id", "text"])
    >>> print(list(ds))
    [["1", "text 1"], ["2", "text 2"]]

    Attributes
    ----------
    df : pd.DataFrame
        a pandas DataFrame object
    columns : Union[str, list]
        a list of column names or a single column name. This value
        is used to select data from those columns only
    """

    def __init__(self, df, columns: Union[str, List[str]]) -> None:
        super().__init__(applied_ops=None, items=df)
        self.columns = columns

    def __iter__(self):
        for i, row in self.items.iterrows():
            if isinstance(self.columns, list):
                yield [row[c] for c in self.columns]
            else:
                yield row[self.columns]


class CSVDataStream(DataFrameStream):
    """Represents a stream of data by reading the contents from a csv file.
    pandas library is used to read the csv.

    Example
    -------
    >>> ds = CSVDataStream(path="news_articles.csv", columns=["body", "title"])

    Attributes
    ----------
    df : pd.DataFrame
        a pandas DataFrame object created after reading the csv file
    columns : Union[str, list]
        a list of column names or a single column name. This value is used to
        select data from those columns only
    path : str
        path to the csv file
    """

    def __init__(self, path: str, columns: Union[str, List[str]]):
        super().__init__(pd.read_csv(path), columns=columns)
        self.path = path
