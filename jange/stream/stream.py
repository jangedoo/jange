from typing import List, Optional, Union
import pandas as pd
from jange.base import DataStream

__all__ = ["DataFrameStream", "CSVDataStream", "from_csv", "from_df"]


class DataFrameStream(DataStream):
    """Represents a stream of data by iterating over the rows in a
    pandas DataFrame object.

    Parameters
    ----------
    df : pd.DataFrame
        pandas DataFrame object

    columns : Union[str, List[str]]
        a column name or a list of column names in the dataframe. The values
        from the given column(s) are used to create a stream. If a list is
        passed then each item in the stream will be a list of values for the given
        columns in that order.

    Example
    -------
    >>> df = pd.DataFrame([{"text": "text 1", "id": "1"}, {"text": "text 2", "id": "2"}])
    >>> ds = DataFrameStream(df=df, columns="text")
    >>> print(list(ds))
    >>> ["text 1", "text 2"]
    >>> ds = DataFrameStream(df=df, columns=["id", "text"])
    >>> print(list(ds))
    >>> [["1", "text 1"], ["2", "text 2"]]

    Attributes
    ----------
    df : pd.DataFrame
        a pandas DataFrame object
    columns : Union[str, list]
        a list of column names or a single column name. This value
        is used to select data from those columns only
    """

    def __init__(
        self, df, columns: Union[str, List[str]], context_column: Optional[str] = None
    ) -> None:
        super().__init__(applied_ops=None, items=["dummy"])
        self.context, self.items = self._get_items(df, columns, context_column)
        self.columns = columns
        self.context_column = context_column

    def _get_items(self, df, columns, context_column):
        contexts = []
        items = []
        for i, row in df.iterrows():
            context = row[context_column] if context_column else i
            contexts.append(context)
            if isinstance(columns, list):
                value = [row[c] for c in columns]
            else:
                value = row[columns]

            items.append(value)

        return contexts, items


class CSVDataStream(DataFrameStream):
    """Represents a stream of data by reading the contents from a csv file.
    pandas library is used to read the csv.

    Parameters
    ----------
    path : str
        path to the csv file to read. This parameter is passed directly to
        `pandas.read_csv` method

    columns : Union[str, List[str]]
        a column name or a list of column names in the csv file. The values
        from the given column(s) are used to create a stream. If a list is
        passed then each item in the stream will be a list of values for the given
        columns in that order.

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

    def __init__(
        self,
        path: str,
        columns: Union[str, List[str]],
        context_column: Optional[str] = None,
    ):
        super().__init__(
            pd.read_csv(path), columns=columns, context_column=context_column
        )
        self.path = path


def from_csv(path: str, columns: list) -> DataStream:
    return CSVDataStream(path=path, columns=columns)


def from_df(df, columns: list) -> DataStream:
    return DataFrameStream(df, columns)
