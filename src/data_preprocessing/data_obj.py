import pandas as pd


class Data:
    def __init__(
        self,
        df: pd.DataFrame = None,
        name: str = "train.csv",
        target_col_name: str = None,
    ):
        if df is None:
            self._df = self.load_data(name)
        else:
            self._df = df
        if target_col_name is not None:
            self._target = self._df[target_col_name]
            self._features = self._df.drop(target_col_name, axis=1)
        else:
            self._target = None
            self._features = self._df.copy()

    @property
    def features(self):
        return self._features

    @property
    def target(self):
        return self._target

    def load_data(name):
        dir = rf"data/{name}"
        if name.endswith("csv"):
            return pd.read_csv(dir)
        else:
            return pd.read_pickle(dir)
