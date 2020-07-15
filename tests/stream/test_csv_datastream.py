from unittest.mock import patch, Mock
import pandas as pd
import pytest
from jange.stream import CSVDataStream, from_csv


@pytest.fixture
def df():
    return pd.DataFrame([{"text": "text 1", "id": "1"}, {"text": "text 2", "id": "2"}])


@patch("jange.stream.stream.pd.read_csv")
def test_can_iterate_all_rows(read_csv: Mock, df):
    read_csv.return_value = df
    ds = CSVDataStream(path="test.csv", columns="text")
    assert len(list(ds)) == len(df)


@pytest.mark.parametrize(
    "columns,expected",
    [
        ("text", ["text 1", "text 2"]),
        ("id", ["1", "2"]),
        (["text"], [["text 1"], ["text 2"]]),
        (["text", "id"], [["text 1", "1"], ["text 2", "2"]]),
    ],
)
@patch("jange.stream.stream.pd.read_csv")
def test_only_returns_data_for_selected_columns(read_csv, df, columns, expected):
    read_csv.return_value = df
    ds = CSVDataStream(path="test.csv", columns=columns)
    assert list(ds) == expected


@patch("jange.stream.stream.pd.read_csv")
def test_from_csv_returns_csv_stream(read_csv):
    path = "test.csv"
    columns = "text"
    ds = from_csv(path="test.csv", columns="text")
    assert isinstance(ds, CSVDataStream)
    assert ds.path == path
    assert ds.columns == columns
