from data_obj import Data
from handling_nan import handle_nan
from handling_no_variance import handle_zero_variance
from transforming_data import yeo_johnson_transform

# TODO when test set is not separated and is supposed to be splitted from dataset (maybe adding test and train to Data obj)
# TODO transformation of data


def preprocess(train_name, test_name, target_col="TARGET"):
    train_obj = Data(df=None, name=train_name, target_col_name=target_col)
    test_obj = Data(df=None, name=test_name)
    no_var_feature_class_dic = handle_zero_variance(train_obj, test_obj)
    removed_columns_info = handle_nan(train_obj, test_obj, nan_threshold=None)
    yeo_johnson_transform(train_obj, test_obj)


preprocess(train_name="train.csv", test_name="test.csv", target_col="TARGET")
