import pandas as pd
import requests

from src.config import DROP_COLUMNS, TARGET, VAL_PATH

URL = "http://127.0.0.1:8000"


def send(nume, features):
    print(f"\nTrimitem: {nume}")
    response = requests.post(f"{URL}/predict", json={"features": features})
    print("Status:", response.status_code)
    print("Raspuns:", response.json())


def main():
    health = requests.get(f"{URL}/health")
    print("Stare API:", health.json())

    df = pd.read_csv(VAL_PATH)
    fraud_row = df[df[TARGET] == 1].iloc[0].drop(labels=DROP_COLUMNS)
    normal_row = df[df[TARGET] == 0].iloc[0].drop(labels=DROP_COLUMNS)

    send("tranzactie frauduloasa", fraud_row.to_dict())
    send("tranzactie normala", normal_row.to_dict())
    send("tranzactie cu feature-uri lipsa", {"V1": 0.0})


if __name__ == "__main__":
    main()
