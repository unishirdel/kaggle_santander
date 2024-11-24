import pandas as pd

from src.data_preprocessing.data_obj import Data


class Preprocessor:
    def __init__(self, train_name, test_name, target_col="TARGET"):
        self.train = Data(train_name, target_col)
        self.test = Data(test_name)

    def remove_zero_variance_columns(self):
        selected_col = []
        for col_name in self.train._features.columns:
            self.add_zero_variance_col(selected_col, col_name)
        self.train.remove_columns(selected_col)
        self.test.remove_columns(selected_col)

    def add_zero_variance_col(self, selected_col, col_name):
        col = self._features[col_name]
        if col.var() == 0:
            return selected_col.append(col_name)
        else:
            return selected_col
