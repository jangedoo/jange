import pandas as pd
import pytest
from jange.stream import DataFrameStream, from_df


@pytest.fixture
def df():
    return pd.DataFrame([{"text": "text 1", "id": "1"}, {"text": "text 2", "id": "2"}])


def test_can_iterate_all_rows(df):
    ds = DataFrameStream(df=df, columns="text")
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
def test_only_returns_data_for_selected_columns(df, columns, expected):
    ds = DataFrameStream(df=df, columns=columns)
    assert list(ds) == expected


def test_from_df_returns_dataframe_stream():
    df = pd.DataFrame([])
    ds = from_df(df=df, columns="test")
    assert isinstance(ds, DataFrameStream)
