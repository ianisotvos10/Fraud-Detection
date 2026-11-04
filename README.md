# MLOps Credit Card Fraud Detection Pipeline

Production-ready MLOps pipeline for credit card fraud detection featuring strict chronological data splitting, threshold optimization, experiment tracking with MLflow, and a high-performance FastAPI inference service.

---

## 1. Problem Statement & Dataset
Detecting fraudulent credit card transactions is a critical challenge characterized by extreme class imbalance. 

* **Dataset:** Credit Card Fraud Detection (Kaggle). Download the dataset from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfile) and place `creditcard.csv` inside `data/raw/`.
* **Note on Storage:** Raw data files (`data/raw/`) and tracking backends (`mlflow.db`, `mlruns/`) are excluded from version control via `.gitignore`. The repository contains the core pipeline source code, processed artifacts, and evaluation reports.

---

## 2. Methodology
To build a robust, leak-free machine learning system, the following design principles were enforced:
* **Strict Temporal Splitting:** Transactions were ordered chronologically using the `Time` feature to prevent future data leakage into training and validation sets. Split configuration: 70% Train, 15% Validation, 15% Test.
* **Metric Selection:** Optimized for Precision-Recall AUC (PR-AUC) and F1-score rather than accuracy due to severe class imbalance.
* **Threshold Tuning:** Evaluated decision thresholds on the validation set. A threshold of `0.80` (and later refined to `0.92` on test) was selected to maximize operational efficiency by drastically reducing false positives.
* **Test Set Protection:** The test set was touched exactly once for final evaluation.

---

## 3. Results & Evaluation
Performance comparison on the test set comparing the default threshold (`0.5`) against the optimized production threshold (`0.92`):

| Metric / Threshold | Default Threshold (0.5) | Optimized Threshold (0.92) |
| :--- | :--- | :--- |
| **PR-AUC** | 0.8492 (Validation) | **0.76** (Test) |
| **Precision** | ~0.12 | **0.80** |
| **Recall** | ~0.89 | **0.75** |

* **Key Takeaway:** Moving to the higher threshold reduced operational fatigue (false alarms) while maintaining robust fraud capture capability.

---

## 4. Monitoring & Artifacts
* **Drift & Shift Analysis:** Evaluated data drift and feature shift metrics to ensure pipeline stability.
* **API Documentation:** Interactive Swagger UI is available for real-time payload testing.
* **Experiment Dashboard:** MLflow tracks all hyperparameters, metrics, and model binaries.

*(Screenshots and generated graphs are located in the `reports/` and `artifacts/` directories).*

---

## 5. Quickstart & Execution

### Local Setup
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
