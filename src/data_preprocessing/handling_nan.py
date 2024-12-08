import pandas as pd


def handle_nan(train_obj, nan_threshold=None):
    removed_columns_info = handle_nan_numeric_columns(train_obj, nan_threshold)

    return removed_columns_info


def handle_nan_numeric_columns(train_obj, nan_threshold=None):
    numeric_columns = train_obj.numeric_columns
    columns_to_keep, removed_columns_info = remove_high_nan_features(
        train_obj, numeric_columns, nan_threshold
    )
    for column in columns_to_keep:
        if train_obj._features[column].isnull().any():
            nan_indices = train_obj._features[
                train_obj._features[column].isna()
            ].index.tolist()
            median_value = train_obj._features[column].median()
            train_obj.set_value(column, nan_indices, median_value)
    return removed_columns_info


def remove_high_nan_features(train_obj, columns=None, nan_threshold=None):
    if columns is None:
        columns = train_obj._features.columns
    nan_proportions = train_obj._features.isna().mean()
    if nan_threshold is not None:
        columns_to_remove = nan_proportions[nan_proportions > nan_threshold].index
        columns_to_keep = nan_proportions[nan_proportions <= nan_threshold].index
    else:
        columns_to_remove = []
        columns_to_keep = columns
    train_obj.remove_columns(columns_to_remove)
    removed_columns_info = pd.DataFrame(
        {
            "column": columns_to_remove,
            "nan_proportion": nan_proportions[columns_to_remove],
        }
    )
    remove_zero_variance_columns()
    return columns_to_keep, removed_columns_info
