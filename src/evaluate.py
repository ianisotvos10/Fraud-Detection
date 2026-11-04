import mlflow
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.config import MODEL_EXPORT_PATH, TARGET_PRECISION, TEST_PATH, THRESHOLD_PATH, VAL_PATH
from src.features import load_xy
from src.tracking import export_model, load_latest_model, save_threshold, set_tracking


def choose_threshold(y_val, proba):
    precision, recall, thresholds = precision_recall_curve(y_val, proba)
    precision = precision[:-1]
    recall = recall[:-1]

    candidates = np.where(precision >= TARGET_PRECISION)[0]
    if len(candidates) > 0:
        best = candidates[np.argmax(recall[candidates])]
        criteriu = f"recall maxim cu precizie >= {TARGET_PRECISION:.2f}"
    else:
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        best = int(np.argmax(f1))
        criteriu = "F1 maxim (nicio valoare nu atinge precizia tinta)"

    return float(thresholds[best]), criteriu


def compute_report(nume, y, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    metrics = {
        "pr_auc": float(average_precision_score(y, proba)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }

    print(f"\n--- Rezultate pe {nume} (prag {threshold:.4f}) ---")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Precizie: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
    print(f" -> Fraude prinse (TP): {tp}")
    print(f" -> Fraude ratate (FN): {fn}")
    print(f" -> Alarme false (FP): {fp}")
    return metrics


def evaluate():
    print("Incarcam modelul si datele...")
    model, run_id = load_latest_model()
    X_val, y_val = load_xy(VAL_PATH)
    X_test, y_test = load_xy(TEST_PATH)

    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    threshold, criteriu = choose_threshold(y_val, val_proba)
    print(f"Prag ales pe validare: {threshold:.4f}")
    print(f"Criteriu: {criteriu}")

    compute_report("validare", y_val, val_proba, threshold)
    test_metrics = compute_report("test", y_test, test_proba, threshold)

    save_threshold(threshold, run_id, criteriu)
    export_model(model)
    print(f"\nPragul a fost salvat in {THRESHOLD_PATH}")
    print(f"Modelul pentru servire a fost exportat in {MODEL_EXPORT_PATH}")

    set_tracking()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("threshold", threshold)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    print("Setul de test a fost folosit o singura data. Nu mai modifica modelul dupa aceasta rulare.")


if __name__ == "__main__":
    evaluate()
