import pandas as pd


def load_data(name):
    return pd.read_csv(rf"data/{name}.csv")
