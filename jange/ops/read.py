from jange.stream import DataFrameStream
from jange.stream import DataStream
from jange.stream import CSVDataStream
from .base import Operation


def csv(path: str, columns: list) -> DataStream:
    return CSVDataStream(path=path, columns=columns)


def df(df, columns: list) -> DataStream:
    return DataFrameStream(df, columns)
