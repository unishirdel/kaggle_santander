import pandas as pd


class Preprocessor:
    def __init__(self, df: pd.DataFrame, target_col_name: str):
        self._target = df[target_col_name]
        self._features = df.drop(target_col_name, axis=1)

    @property
    def features(self):
        return self._features

    @property
    def target(self):
        return self._target

    def remove_zero_variance_columns(self):
        selected_col = []
        for col_name in self._features.columns:
            self.add_non_zero_variance_col(selected_col, col_name)
        return self._features[selected_col]

    def add_non_zero_variance_col(self, selected_col, col_name):
        col = self._features[col_name]
        if col.var() != 0:
            return selected_col.append(col_name)
        else:
            return selected_col


Preprocessor(
    df=pd.read_csv("data/train.csv"), target_col_name="TARGET"
).remove_zero_variance_columns()
