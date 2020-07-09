from unittest.mock import patch
import pandas as pd
from jange import ops, stream


@patch("jange.stream.stream.pd.read_csv")
def test_read_csv_returns_csv_stream(read_csv):
    path = "test.csv"
    columns = "text"
    ds = ops.read.csv(path="test.csv", columns="text")
    assert isinstance(ds, stream.CSVDataStream)
    assert ds.path == path
    assert ds.columns == columns


def test_df_returns_dataframe_stream():
    df = pd.DataFrame([])
    ds = ops.read.df(df=df, columns="test")
    assert isinstance(ds, stream.DataFrameStream)
