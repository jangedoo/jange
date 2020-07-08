from jange.stream import DataFrameStream
from jange.stream import DataStream
from jange.stream import CSVDataStream

__all__ = ["csv", "df"]


def csv(path: str, columns: list) -> DataStream:
    return CSVDataStream(path=path, columns=columns)


def df(df, columns: list) -> DataStream:
    return DataFrameStream(df, columns)
