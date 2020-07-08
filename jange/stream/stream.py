import pandas as pd


class DataStream:
    def __init__(self, applied_ops: list, items):
        self.applied_ops = applied_ops or []
        self.items = items

    def __iter__(self):
        for i in self.items:
            yield i

    def apply(self, *ops):
        x = self
        for op in ops:
            x = op.run(x)
        return x


class DataFrameStream(DataStream):
    def __init__(self, df, columns: list) -> None:
        super().__init__(applied_ops=None, items=None)
        self.df = df
        self.columns = columns

    def __iter__(self):
        for i, row in self.df.iterrows():
            if isinstance(self.columns, list):
                yield [row[c] for c in self.columns]
            else:
                yield row[self.columns]


class CSVDataStream(DataFrameStream):
    def __init__(self, path: str, columns: list):
        super().__init__(pd.read_csv(self.path), columns=columns)
        self.path = path
