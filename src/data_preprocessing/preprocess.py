from data_obj import Data
from handling_nan import handle_nan
from handling_no_variance import handle_zero_variance

# TODO when test set is not separated and is supposed to be splitted from dataset


def preprocess(train_name, test_name, target_col="TARGET"):
    train_obj = Data(df=None, name=train_name, target_col_name=target_col)
    test_obj = Data(df=None, name=test_name)
    no_var_feature_class_dic = handle_zero_variance(train_obj, test_obj)
    handle_nan(train_obj, test_obj, nan_threshold=None)
