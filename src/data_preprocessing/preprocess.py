import pandas as pd
from data_obj import Data


# TODO handle nan values of numeric columns with other methods instead of median
# TODO when test set is not separated and is supposed to be splitted from dataset
class Preprocessor:
    def __init__(self, train_name, test_name, target_col="TARGET"):
        self.train = Data(df=None, name=train_name, target_col_name=target_col)
        self.test = Data(df=None, name=test_name)

    def handle_nan(self, nan_threshold=None):
        numeric_columns = self.train.numeric_columns
        columns_to_keep, removed_columns_info = self.remove_high_nan_features(
            nan_threshold
        )

        for column in columns_to_keep:
            if self.train._features[column].isnull().any():
                nan_indices = self.train._features[
                    self.train._features[column].isna()
                ].index.tolist()
                median_value = self.train._features[column].median()
                self.train.set_value(column, nan_indices, median_value)

        return removed_columns_info

    def remove_high_nan_features(self, nan_threshold=None):
        columns = self.train._features.columns
        nan_proportions = self.train._features.isna().mean()
        if nan_threshold is not None:
            columns_to_remove = nan_proportions[nan_proportions > nan_threshold].index
            columns_to_keep = nan_proportions[nan_proportions <= nan_threshold].index
        else:
            columns_to_remove = []
            columns_to_keep = columns
        self.train.remove_columns(columns_to_remove)
        removed_columns_info = pd.DataFrame(
            {
                "column": columns_to_remove,
                "nan_proportion": nan_proportions[columns_to_remove],
            }
        )
        return columns_to_keep, removed_columns_info

    def remove_zero_variance_columns(self):
        no_variance_col = []
        for col_name in self.train._features.columns:
            self.add_zero_variance_col(no_variance_col, col_name)
        self.train.remove_columns(no_variance_col)
        self.test.remove_columns(no_variance_col)

    def add_zero_variance_col(self, selected_col, col_name, sample_ind=None):
        if sample_ind is None:
            sample_ind = self.train._features.index
        col = self.train._features[col_name].loc[sample_ind]
        if col.var() == 0:
            return selected_col.append(col_name)
        else:
            return selected_col

    def store_class_zero_variance_cols(self):
        pos_samples, neg_samples = self.train.get_class_samples()
        no_var_col_pos, no_var_col_neg = [], []
        for col_name in self.train._features.columns:
            self.add_zero_variance_col(no_var_col_pos, col_name, pos_samples.index)
            self.add_zero_variance_col(no_var_col_neg, col_name, neg_samples.index)
        no_var_feature_class_dic = {}
        no_var_feature_class_dic = self.create_no_var_feature_class_dic(
            no_var_feature_class_dic, no_var_col_pos, cl=1
        )
        no_var_feature_class_dic = self.create_no_var_feature_class_dic(
            no_var_feature_class_dic, no_var_col_neg, cl=0
        )
        return no_var_feature_class_dic

    def create_no_var_feature_class_dic(self, dic, no_var_col, cl):
        for col_name in no_var_col:
            feature = self.train._features[col_name]
            target = self.train._target
            no_var_val = feature[target == cl].iloc[0]
            dic[col_name] = (no_var_val, cl)
        return dic
