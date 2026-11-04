import os

import pandas as pd

from src.config import PROCESSED_DIR, RAW_PATH, TARGET


def time_based_split(train_ratio=0.7, val_ratio=0.15):
    print("Incarcam dataset-ul brut...")
    df = pd.read_csv(RAW_PATH)

    if "Time" not in df.columns:
        raise ValueError("Coloana Time lipseste. Verifica dataset-ul.")

    df = df.sort_values("Time").reset_index(drop=True)
    n_samples = len(df)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    parts = {
        "train": df.iloc[:train_end],
        "val": df.iloc[train_end:val_end],
        "test": df.iloc[val_end:],
    }

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print("Salvam partitiile...")
    for name, part in parts.items():
        part.to_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"), index=False)
        procent = len(part) / n_samples * 100
        print(f"{name}: {len(part)} randuri ({procent:.1f}%), fraude: {int(part[TARGET].sum())}")

    print("Split realizat cu succes.")


if __name__ == "__main__":
    time_based_split()
