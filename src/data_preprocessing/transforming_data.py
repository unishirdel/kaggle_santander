from data_obj import Data
from sklearn.preprocessing import PowerTransformer


def yeo_johnson_transform(train_obj: Data, test_obj: Data):
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    numeric_columns = train_obj.numeric_columns
    pt.fit(train_obj._features[numeric_columns])
    train_obj.transform_numeric_columns(pt)
    test_obj.transform_numeric_columns(pt)
