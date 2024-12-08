import pandas as pd


# TODO how to make the attributes created by set_attribute function private
class Data:
    dir_ = "data"

    def __init__(
        self,
        df: pd.DataFrame = None,
        name: str = "train.csv",
        target_col_name: str = None,
        id_columns: list[str] = None,
    ):
        if df is None:
            self._df = self.load_data(name)
        else:
            self._df = df

        if id_columns is None:
            id_columns = [col for col in self._df.columns if "id" in col.lower()]

        if target_col_name is not None:
            self._target = self._df[target_col_name]
            self._features = self._df.drop([target_col_name] + id_columns, axis=1)
        else:
            self._target = None
            self._features = self._df.copy()
            self._features = self._df.drop(id_columns, axis=1)

        columns_data_type = self.get_columns_data_type()
        self.set_attributes(columns_data_type)

    @property
    def features(self):
        return self._features

    @property
    def target(self):
        return self._target

    def set_attributes(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

    def concat_features_target(self):
        if self._target is None:
            df = self._features
        else:
            df = pd.concat([self._features, self._target], axis=1)
        return df

    def remove_columns(self, col_names: list):
        self._features = self._features.drop(columns=col_names)
        columns_data_type = self.get_columns_data_type()
        self.set_attributes(columns_data_type)

    def get_class_samples(self):
        pos_samples = self._features[self._target == 1]
        neg_samples = self._features[self._target == 0]
        return pos_samples, neg_samples

    def get_columns_data_type(self):
        columns_data_type = {}
        columns_data_type["numeric_columns"] = self._features.select_dtypes(
            include=["number"]
        ).columns.tolist()
        columns_data_type["boolean_columns"] = self._features.select_dtypes(
            include=["bool"]
        ).columns.tolist()
        columns_data_type["categorical_columns"] = self._features.select_dtypes(
            include=["object"]
        ).columns.tolist()
        return columns_data_type

    def set_value(self, column_name, index, value):
        self._features.loc[index, column_name] = value

    def load_data(self, name):
        dir_ = Data.dir_ + "/" + name
        if name.endswith("csv"):
            return pd.read_csv(dir_)
        else:
            return pd.read_pickle(dir_)

    def save_data(self, name):
        dir_ = Data.dir_ + "/" + name
        df = self.concat_features_target()
        if name.endswith(".csv"):
            return df.to_csv(dir_)
        else:
            return df.to_pickle(dir_)
