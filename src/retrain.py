import argparse
import os

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_score, recall_score

from src.config import REPORTS_DIR, TEST_PATH, TRAIN_PATH, VAL_PATH
from src.drift import SHIFTED_FEATURES, build_batches, get_top_features
from src.evaluate import choose_threshold
from src.features import load_xy
from src.tracking import (
    export_model,
    load_serving_model,
    load_threshold,
    promote_run,
    save_threshold,
    set_stage,
    set_tracking,
    setup_experiment,
)
from src.train import PARAMS

ADAPT_RATIO = 0.5
MIN_HOLDOUT_FRAUDS = 5
PROMOTION_MARGIN = 0.02
DEFAULT_ROUNDS = 150


def read_alerted_batches(scenario):
    path = os.path.join(REPORTS_DIR, f"drift_report_{scenario}.csv")
    if not os.path.exists(path):
        return None
    report = pd.read_csv(path)
    return report.loc[report["alert"], "batch"].astype(int).tolist()


def champion_rounds(run_id):
    if run_id is None:
        return DEFAULT_ROUNDS
    params = mlflow.get_run(run_id).data.params
    if "best_iteration" in params:
        return int(float(params["best_iteration"])) + 1
    return DEFAULT_ROUNDS


def metrics_at(y, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
    }


def retrain(scenario):
    alerted = read_alerted_batches(scenario)
    if alerted is None:
        print(f"Raportul pentru scenariul {scenario} nu exista. Ruleaza: python -m src.monitor --scenario {scenario}")
        return
    if not alerted:
        print("Nu exista alerta de drift. Reantrenarea nu este necesara.")
        return

    first_alert = alerted[0]
    print(f"Alerta de drift incepand cu lotul {first_alert}. Pornim reantrenarea...")

    set_tracking()
    old_model, old_run_id = load_serving_model()
    old_threshold = load_threshold(old_run_id)
    rounds = champion_rounds(old_run_id)

    X_train, y_train = load_xy(TRAIN_PATH)
    X_val, y_val = load_xy(VAL_PATH)
    stream_df = pd.read_csv(TEST_PATH).sort_values("Time").reset_index(drop=True)

    top = get_top_features(old_model, X_val.columns)
    batches = build_batches(stream_df, scenario, top[:SHIFTED_FEATURES], X_val.std())
    recent = [(X, y) for number, X, y in batches if number >= first_alert]
    X_recent = pd.concat([X for X, _ in recent], ignore_index=True)
    y_recent = np.concatenate([y for _, y in recent])

    cut = int(len(X_recent) * ADAPT_RATIO)
    X_adapt, y_adapt = X_recent.iloc[:cut], y_recent[:cut]
    X_hold, y_hold = X_recent.iloc[cut:], y_recent[cut:]
    holdout_frauds = int(y_hold.sum())

    print(f"Fereastra recenta: {len(X_recent)} randuri")
    print(f"Adaptare: {len(X_adapt)} randuri | Evaluare: {len(X_hold)} randuri, fraude: {holdout_frauds}")

    if holdout_frauds < MIN_HOLDOUT_FRAUDS:
        print(f"Prea putine fraude in setul de evaluare (minim {MIN_HOLDOUT_FRAUDS}). Pastram modelul curent.")
        return

    X_new = pd.concat([X_train, X_val, X_adapt], ignore_index=True)
    y_new = np.concatenate([y_train.to_numpy(), y_val.to_numpy(), y_adapt])
    scale_weight = float((y_new == 0).sum() / (y_new == 1).sum())
    params = {**PARAMS, "n_estimators": rounds}

    print("Antrenam modelul candidat...")
    candidate = xgb.XGBClassifier(
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric="aucpr",
        **params,
    )
    candidate.fit(X_new, y_new, verbose=False)

    old_proba = old_model.predict_proba(X_hold)[:, 1]
    new_proba = candidate.predict_proba(X_hold)[:, 1]
    old_pr = float(average_precision_score(y_hold, old_proba))
    new_pr = float(average_precision_score(y_hold, new_proba))
    old_at = metrics_at(y_hold, old_proba, old_threshold)
    new_at = metrics_at(y_hold, new_proba, old_threshold)

    print("\n--- Comparatie pe fereastra de evaluare (acelasi prag pentru ambele) ---")
    print(f"Model curent:   PR-AUC {old_pr:.4f} | Precizie {old_at['precision']:.4f} | Recall {old_at['recall']:.4f}")
    print(f"Model candidat: PR-AUC {new_pr:.4f} | Precizie {new_at['precision']:.4f} | Recall {new_at['recall']:.4f}")

    promote = new_pr >= old_pr + PROMOTION_MARGIN

    setup_experiment()
    with mlflow.start_run(run_name="retrain") as run:
        mlflow.set_tags(
            {
                "stage": "candidate",
                "parent_run": str(old_run_id),
                "trigger": f"alerta_drift_{scenario}",
            }
        )
        mlflow.log_params(
            {
                **params,
                "scale_pos_weight": scale_weight,
                "best_iteration": rounds - 1,
                "adapt_rows": len(X_adapt),
            }
        )
        mlflow.log_metrics(
            {
                "holdout_pr_auc_candidate": new_pr,
                "holdout_pr_auc_champion": old_pr,
                "holdout_frauds": holdout_frauds,
            }
        )
        mlflow.xgboost.log_model(candidate, "model")
        new_run_id = run.info.run_id

    if promote:
        new_threshold, criteriu = choose_threshold(y_hold, new_proba)
        promote_run(new_run_id)
        save_threshold(new_threshold, new_run_id, criteriu + " (ales pe fereastra recenta, de reconfirmat cu date noi)")
        export_model(candidate)
        print(f"\nModelul candidat a fost promovat (run {new_run_id}).")
        print(f"Prag nou: {new_threshold:.4f}, ales pe fereastra recenta, deci de reconfirmat cu date noi.")
        print("Reporneste API-ul pentru a incarca modelul nou.")
    else:
        set_stage(new_run_id, "rejected")
        print(f"\nCandidatul nu depaseste modelul curent cu cel putin {PROMOTION_MARGIN:.2f} PR-AUC. Pastram modelul curent.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["natural", "shift"], default="shift")
    args = parser.parse_args()
    retrain(args.scenario)
