import pandas as pd
from data_obj import Data

# TODO Handling categorical nan value (eiher by new category or by making prediction for the categories of these samples)
# TODO Handling boolean cnan values (either similar to categorical columns, or consider it ordinal feature and use the average for nan)
# TODO handle nan values of numeric columns with other methods instead of median


def handle_nan(train_obj: Data, test_obj: Data, nan_threshold=None):
    removed_columns_info = remove_high_nan_features(train_obj, test_obj, nan_threshold)
    handle_nan_numeric_columns(train_obj, test_obj)
    return removed_columns_info


def remove_high_nan_features(train_obj: Data, test_obj: Data, nan_threshold=None):
    columns_to_remove, removed_columns_info = find_columns_to_remove(
        train_obj, nan_threshold
    )
    train_obj.remove_columns(columns_to_remove)
    test_obj.remove_columns(columns_to_remove)
    return removed_columns_info


def find_columns_to_remove(train_obj, nan_threshold):
    nan_proportions = train_obj._features.isna().mean()
    if nan_threshold is not None:
        columns_to_remove = nan_proportions[nan_proportions > nan_threshold].index
    else:
        columns_to_remove = []
    removed_columns_info = pd.DataFrame(
        {
            "column": columns_to_remove,
            "nan_proportion": nan_proportions[columns_to_remove],
        }
    )
    return columns_to_remove, removed_columns_info


def handle_nan_numeric_columns(train_obj: Data, test_obj: Data):
    numeric_columns = train_obj.numeric_columns
    for column in numeric_columns:
        median_value = train_obj._features[column].median()
        set_nan_numeric_columns(train_obj, column, median_value)
        set_nan_numeric_columns(test_obj, column, median_value)


def set_nan_numeric_columns(data_obj, column, median_value):
    nan_indices = data_obj._features[data_obj._features[column].isna()].index.tolist()
    data_obj.set_value(column, nan_indices, median_value)


def handle_nan_categorical_columns():
    pass


def handle_nan_boolean_columns():
    pass
