import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import TRAIN_PATH, VAL_PATH
from src.features import load_xy
from src.tracking import promote_run, setup_experiment

PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
EARLY_STOPPING_ROUNDS = 20


def train_baseline():
    print("Incarcam datele partitionate...")
    X_train, y_train = load_xy(TRAIN_PATH)
    X_val, y_val = load_xy(VAL_PATH)

    scale_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    setup_experiment()

    with mlflow.start_run() as run:
        print("Antrenam modelul XGBoost...")
        clf = xgb.XGBClassifier(
            scale_pos_weight=scale_weight,
            random_state=42,
            eval_metric="aucpr",
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            **PARAMS,
        )
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        print("Calculam metricile pe validare...")
        proba = clf.predict_proba(X_val)[:, 1]
        pred = (proba >= 0.5).astype(int)

        pr_auc = average_precision_score(y_val, proba)
        precision = precision_score(y_val, pred, zero_division=0)
        recall = recall_score(y_val, pred)
        f1 = f1_score(y_val, pred)
        cm = confusion_matrix(y_val, pred)

        mlflow.log_params(
            {
                **PARAMS,
                "scale_pos_weight": scale_weight,
                "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                "best_iteration": clf.best_iteration,
            }
        )
        mlflow.log_metrics(
            {
                "val_pr_auc": pr_auc,
                "val_precision_at_0.5": precision,
                "val_recall_at_0.5": recall,
                "val_f1_at_0.5": f1,
            }
        )
        mlflow.xgboost.log_model(clf, "model")
        promote_run(run.info.run_id)

        print("Antrenare finalizata cu succes.")
        print(f"Iteratia optima: {clf.best_iteration}")
        print(f"PR-AUC: {pr_auc:.4f} | F1 la prag 0.50: {f1:.4f}")
        print(f"Precizie: {precision:.4f} (cat de sigure sunt alertele)")
        print(f"Recall: {recall:.4f} (cate fraude din total sunt prinse)")
        print(f"Matricea de confuzie:\n{cm}")
        print("Modelul a fost marcat ca model de productie. Ruleaza: python -m src.evaluate")


if __name__ == "__main__":
    train_baseline()
