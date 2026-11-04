import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
VAL_PATH = os.path.join(PROCESSED_DIR, "val.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")

MLFLOW_DB = os.path.join(BASE_DIR, "mlflow.db").replace("\\", "/")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
THRESHOLD_PATH = os.path.join(ARTIFACTS_DIR, "threshold.json")
MODEL_EXPORT_PATH = os.path.join(ARTIFACTS_DIR, "model.json")

EXPERIMENT_NAME = "fraud_detection_xgb"
TARGET = "Class"
DROP_COLUMNS = ["Class", "Time"]
TARGET_PRECISION = 0.90
