import pandas as pd

from src.config import DROP_COLUMNS, TARGET


def load_xy(path):
    df = pd.read_csv(path)
    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET]
    return X, y
