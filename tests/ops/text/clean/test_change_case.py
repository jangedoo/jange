import pytest
from jange.ops.text import CaseChangeOperation, uppercase, lowercase
from jange.stream import DataStream


@pytest.mark.parametrize(
    "fn,expected_mode", [(uppercase, "upper"), (lowercase, "lower")]
)
def test_helper_fns_returns_correct_operation(fn, expected_mode):
    op: CaseChangeOperation = fn()
    assert isinstance(op, CaseChangeOperation)
    assert op.mode == expected_mode


def test_raises_error_when_invalid_mode():
    with pytest.raises(ValueError):
        CaseChangeOperation(mode="invalid mode")


@pytest.fixture
def ds():
    return DataStream(items=["Aa", "bbB"])


@pytest.mark.parametrize(
    "mode,expected",
    [("upper", ["AA", "BBB"]), ("lower", ["aa", "bbb"]), ("capitalize", ["Aa", "Bbb"])],
)
def test_correctly_changes_cases(ds: DataStream, mode: str, expected: list):
    op = CaseChangeOperation(mode=mode)
    assert list(ds.apply(op)) == expected
