import numpy as np
import pandas as pd

from src.config import DROP_COLUMNS, TARGET

N_BATCHES = 6
SHIFT_START = N_BATCHES // 2
PSI_LIMIT = 0.2
MIN_DRIFTED_FEATURES = 3
TOP_FEATURES = 10
SHIFTED_FEATURES = 5
SHIFT_STD_MULTIPLIER = 1.5
AMOUNT_MULTIPLIER = 3


def psi(reference, current, bins=10):
    reference = np.asarray(reference)
    current = np.asarray(current)
    quantiles = np.quantile(reference, np.linspace(0, 1, bins + 1))
    inner = np.unique(quantiles[1:-1])
    if len(inner) == 0:
        return 0.0
    ref_counts = np.bincount(np.searchsorted(inner, reference, side="right"), minlength=len(inner) + 1)
    cur_counts = np.bincount(np.searchsorted(inner, current, side="right"), minlength=len(inner) + 1)
    ref_pct = np.clip(ref_counts / len(reference), 1e-4, None)
    cur_pct = np.clip(cur_counts / len(current), 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def inject_shift(X, features, std):
    X = X.copy()
    for name in features:
        X[name] = X[name] + SHIFT_STD_MULTIPLIER * std[name]
    X["Amount"] = X["Amount"] * AMOUNT_MULTIPLIER
    return X


def get_top_features(model, columns, n=TOP_FEATURES):
    importances = pd.Series(model.feature_importances_, index=list(columns))
    return list(importances.sort_values(ascending=False).head(n).index)


def build_batches(stream_df, scenario, shifted_features, std):
    positions = np.array_split(np.arange(len(stream_df)), N_BATCHES)
    batches = []
    for number, pos in enumerate(positions, start=1):
        batch = stream_df.iloc[pos]
        X = batch.drop(columns=DROP_COLUMNS)
        y = batch[TARGET].to_numpy()
        if scenario == "shift" and number > SHIFT_START:
            X = inject_shift(X, shifted_features, std)
        batches.append((number, X, y))
    return batches
