import pandas as pd
from handling_no_variance import remove_zero_variance_columns

# TODO Handling categorical nan value (eiher by new category or by making prediction for the categories of these samples)
# TODO Handling boolean cnan values (either similar to categorical columns, or consider it ordinal feature and use the average for nan)
# TODO handle nan values of numeric columns with other methods instead of median


def handle_nan(train_obj, test_obj, nan_threshold=None):
    removed_columns_info = handle_nan_numeric_columns(
        train_obj, test_obj, nan_threshold
    )
    return removed_columns_info


def handle_nan_numeric_columns(train_obj, test_obj, nan_threshold=None):
    numeric_columns = train_obj.numeric_columns
    columns_to_keep, removed_columns_info = remove_high_nan_features(
        train_obj, test_obj, numeric_columns, nan_threshold
    )
    for column in columns_to_keep:
        median_value = train_obj._features[column].median()
        set_nan_numeric_columns(train_obj, column, median_value)
        set_nan_numeric_columns(test_obj, column, median_value)
    return removed_columns_info


def remove_high_nan_features(train_obj, test_obj, columns=None, nan_threshold=None):
    columns_to_keep, columns_to_remove, nan_proportions = find_columns_to_remove(
        train_obj, columns, nan_threshold
    )
    train_obj.remove_columns(columns_to_remove)
    test_obj.remove_columns(columns_to_remove)
    removed_columns_info = pd.DataFrame(
        {
            "column": columns_to_remove,
            "nan_proportion": nan_proportions[columns_to_remove],
        }
    )
    return columns_to_keep, removed_columns_info


def find_columns_to_remove(train_obj, columns, nan_threshold):
    if columns is None:
        columns = train_obj._features.columns
    nan_proportions = train_obj._features.isna().mean()
    if nan_threshold is not None:
        columns_to_remove = nan_proportions[nan_proportions > nan_threshold].index
        columns_to_keep = nan_proportions[nan_proportions <= nan_threshold].index
    else:
        columns_to_remove = []
        columns_to_keep = columns
    return columns_to_keep, columns_to_remove, nan_proportions


def set_nan_numeric_columns(data_obj, column, median_value):
    nan_indices = data_obj._features[data_obj._features[column].isna()].index.tolist()
    data_obj.set_value(column, nan_indices, median_value)


def handle_nan_categorical_columns():
    pass


def handle_nan_boolean_columns():
    pass
