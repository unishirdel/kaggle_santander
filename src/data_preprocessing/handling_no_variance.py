def handle_zero_variance(train_obj, test_obj):
    remove_zero_variance_columns(train_obj, test_obj)
    no_var_feature_class_dic = store_class_zero_variance_cols(train_obj)
    return no_var_feature_class_dic


def remove_zero_variance_columns(train_obj, test_obj):
    no_variance_col = []
    for col_name in train_obj._features.columns:
        add_zero_variance_col(train_obj, no_variance_col, col_name)
    train_obj.remove_columns(no_variance_col)
    test_obj.remove_columns(no_variance_col)


def add_zero_variance_col(train_obj, selected_col, col_name, sample_ind=None):
    if sample_ind is None:
        sample_ind = train_obj._features.index
    col = train_obj._features[col_name].loc[sample_ind]
    if col.var() == 0:
        return selected_col.append(col_name)
    else:
        return selected_col


def store_class_zero_variance_cols(train_obj):
    pos_samples, neg_samples = train_obj.get_class_samples()
    no_var_col_pos, no_var_col_neg = [], []
    for col_name in train_obj._features.columns:
        add_zero_variance_col(train_obj, no_var_col_pos, col_name, pos_samples.index)
        add_zero_variance_col(train_obj, no_var_col_neg, col_name, neg_samples.index)
    no_var_feature_class_dic = {}
    no_var_feature_class_dic = create_no_var_feature_class_dic(
        train_obj, no_var_feature_class_dic, no_var_col_pos, cl=1
    )
    no_var_feature_class_dic = create_no_var_feature_class_dic(
        train_obj, no_var_feature_class_dic, no_var_col_neg, cl=0
    )
    return no_var_feature_class_dic


def create_no_var_feature_class_dic(train_obj, dic, no_var_col, cl):
    for col_name in no_var_col:
        feature = train_obj._features[col_name]
        target = train_obj._target
        no_var_val = feature[target == cl].iloc[0]
        dic[col_name] = (no_var_val, cl)
    return dic
