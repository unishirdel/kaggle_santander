import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance


def rank_features(features, target, methods=None, weights=None):
    available_methods = {
        "rf": compute_rf_importance,
        "mi": compute_mutual_information,
        "permutation": compute_permutation_importance,
        "chi": compute_chi_square,
        "iv_woe": compute_iv_woe,
    }

    if methods is None:
        methods = available_methods.keys()

    if weights is None:
        weights = {method: 1.0 for method in methods}

    rankings = pd.DataFrame(index=features.columns)
    for method in methods:
        if method in available_methods:
            func = available_methods[method]
            scores = func(features=features, target=target)
            rankings[method] = scores
            rankings[f"{method}_rank"] = rankings[method].rank(
                ascending=False, method="average"
            )
    rank_columns = [f"{method}_rank" for method in methods]
    method_weights = [weights.get(method, 1.0) for method in methods]
    rankings["average_rank"] = rankings[rank_columns].multiply(
        method_weights, axis=1
    ).sum(axis=1) / sum(method_weights)
    rankings_sorted = rankings.sort_values("average_rank")
    ranking_results = rankings_sorted[rank_columns + ["average_rank"]]
    return ranking_results


def compute_rf_importance(features, target):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(features, target)
    importances = pd.Series(rf.feature_importances_, index=features.columns)
    return importances


def compute_mutual_information(features, target):
    mi_scores = mutual_info_classif(features, target, random_state=42)
    return pd.Series(mi_scores, index=features.columns)


def compute_permutation_importance(features, target, model=None, n_repeats=10):
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, target)
    result = permutation_importance(
        model, features, target, n_repeats=n_repeats, random_state=42, n_jobs=-1
    )
    return pd.Series(result.importances_mean, index=features.columns)


def compute_chi_square(features, target, bins=10):
    chi_scores = pd.Series(index=features.columns, dtype=float)

    for column in features.columns:
        # Bin continuous features
        if features[column].nunique() > bins:
            feature_binned = pd.qcut(
                features[column], bins, labels=False, duplicates="drop"
            )
        else:
            feature_binned = features[column]
        contingency_table = pd.crosstab(feature_binned, target)
        chi2, pvalue, _, _ = chi2_contingency(contingency_table)
        chi_scores[column] = -np.log10(pvalue) if pvalue > 0 else np.inf
    return chi_scores


def compute_iv_woe(features, target, bins=10):
    iv_scores = {}
    woe_dict = {}

    # Ensure binary target is encoded as 0 and 1
    target_values = sorted(target.unique())
    target = target.replace({target_values[0]: 0, target_values[1]: 1})

    for column in features.columns:
        # Handle categorical and continuous features
        if (
            features[column].dtype.kind in {"O", "b", "i", "u"}
            or features[column].nunique() <= bins
        ):
            # Treat as categorical
            feature = features[column].astype(str)
        else:
            # Bin continuous variables
            feature = pd.qcut(features[column], bins, duplicates="drop").astype(str)

        df = pd.DataFrame({"feature": feature, "target": target})
        groups = df.groupby("feature")["target"]

        # Calculate event (bad) and non-event (good) counts
        stats = groups.agg(["count", "sum"])
        stats.columns = ["total", "bad"]
        stats["good"] = stats["total"] - stats["bad"]

        # Replace zeros to avoid division by zero
        stats.replace({"bad": {0: 0.5}, "good": {0: 0.5}}, inplace=True)

        # Calculate distributions
        stats["dist_bad"] = stats["bad"] / stats["bad"].sum()
        stats["dist_good"] = stats["good"] / stats["good"].sum()

        # Calculate WOE
        stats["woe"] = np.log(stats["dist_good"] / stats["dist_bad"])

        # Calculate IV
        stats["iv"] = (stats["dist_good"] - stats["dist_bad"]) * stats["woe"]

        # Total IV for the feature
        iv = stats["iv"].sum()
        iv_scores[column] = iv

        # Store WOE mapping
        woe_mapping = stats["woe"].to_dict()
        woe_dict[column] = woe_mapping

    iv_scores = pd.Series(iv_scores).sort_values(ascending=False)
    return iv_scores
