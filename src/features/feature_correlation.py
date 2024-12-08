import networkx as nx
import numpy as np
import pandas as pd

from data_preprocessing.data_obj import Data


def remove_correlated_features(train_obj: Data, test_obj: Data):
    groups = identify_correlated_groups(train_obj._features, threshold=0.9)
    non_representative_features = find_non_representative_features(
        train_obj._features, groups
    )
    train_obj.remove_columns(non_representative_features)
    test_obj.remove_columns(non_representative_features)


def identify_correlated_groups(data: pd.DataFrame, threshold=0.9):
    corr_matrix = data.corr(method="spearman").abs()
    corr_matrix = corr_matrix.copy()
    np.fill_diagonal(corr_matrix.values, 0)
    adj_matrix = (corr_matrix.abs() >= threshold).astype(int)
    G = nx.from_pandas_adjacency(adj_matrix)
    connected_components = list(nx.connected_components(G))
    return connected_components


def find_non_representative_features(data: pd.DataFrame, groups: list[list]):
    selected_features = []
    for group in groups:
        group = list(group)
        variances = data[group].var()
        representative_feature = variances.idxmax()
        selected_features.append(representative_feature)
    selected_features = set(selected_features)
    all_grouped_features = set().union(*groups)
    non_representative_features = [
        feature for feature in all_grouped_features if feature not in selected_features
    ]
    return non_representative_features
