import json
import os
from pathlib import Path

import mlflow
import mlflow.xgboost
import xgboost as xgb
from mlflow.tracking import MlflowClient

from src.config import (
    ARTIFACTS_DIR,
    EXPERIMENT_NAME,
    MLFLOW_DB,
    MLRUNS_DIR,
    MODEL_EXPORT_PATH,
    THRESHOLD_PATH,
)


def set_tracking():
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")


def setup_experiment():
    set_tracking()
    os.makedirs(MLRUNS_DIR, exist_ok=True)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=Path(MLRUNS_DIR).as_uri(),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)


def search_run(experiment_id, filter_string):
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        return None
    return runs.iloc[0].run_id


def get_production_run_id():
    set_tracking()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError("Experimentul nu exista. Ruleaza mai intai: python -m src.train")

    run_id = search_run(
        experiment.experiment_id,
        "tags.stage = 'production' and attributes.status = 'FINISHED'",
    )
    if run_id is None:
        run_id = search_run(experiment.experiment_id, "attributes.status = 'FINISHED'")
    if run_id is None:
        raise RuntimeError("Nu exista niciun run finalizat. Ruleaza: python -m src.train")
    return run_id


def set_stage(run_id, stage):
    set_tracking()
    MlflowClient().set_tag(run_id, "stage", stage)


def promote_run(run_id):
    set_tracking()
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    previous = client.search_runs(
        [experiment.experiment_id],
        filter_string="tags.stage = 'production'",
    )
    for run in previous:
        if run.info.run_id != run_id:
            client.set_tag(run.info.run_id, "stage", "archived")
    client.set_tag(run_id, "stage", "production")


def load_latest_model():
    run_id = get_production_run_id()
    model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
    return model, run_id


def export_model(model):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model.save_model(MODEL_EXPORT_PATH)


def save_threshold(threshold, run_id, criteriu):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": threshold, "run_id": run_id, "criteriu": criteriu}, f, indent=2)


def load_serving_model():
    if os.path.exists(MODEL_EXPORT_PATH) and os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH) as f:
            data = json.load(f)
        model = xgb.XGBClassifier()
        model.load_model(MODEL_EXPORT_PATH)
        return model, data.get("run_id")
    return load_latest_model()


def load_threshold(run_id=None, default=0.80):
    if not os.path.exists(THRESHOLD_PATH):
        print(f"Fisierul cu pragul lipseste, folosim pragul implicit {default:.2f}. Ruleaza: python -m src.evaluate")
        return default
    with open(THRESHOLD_PATH) as f:
        data = json.load(f)
    if run_id is not None and data.get("run_id") != run_id:
        print("Atentie: pragul salvat apartine altui run. Ruleaza din nou: python -m src.evaluate")
    return float(data["threshold"])
