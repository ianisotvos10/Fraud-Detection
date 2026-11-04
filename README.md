# 💳 MLOps: Credit Card Fraud Detection Pipeline

Salut! Acesta este un proiect de portofoliu prin care am vrut sa depasesc stadiul de "inca un Jupyter Notebook" si sa construiesc un sistem complet de Machine Learning pregatit pentru productie. 

Proiectul nu se concentreaza doar pe antrenarea unui model, ci pe intregul pipeline MLOps: prevenirea data leakage-ului, optimizarea deciziilor pentru un mediu de business (threshold tuning), logarea experimentelor si servirea modelului printr-un API.

## 🛠️ Tech Stack
* **Machine Learning:** Python, XGBoost, scikit-learn, Pandas
* **MLOps & Tracking:** MLflow (cu backend SQLite local)
* **Serving & Backend:** FastAPI, Uvicorn, Pydantic
* **Deployment:** Docker

---

## 🧠 Decizii de Arhitectura (Cum am evitat capcanele clasice)

Cand lucrezi cu date tabulare financiare puternic dezechilibrate, e usor sa obtii o acuratete de 99.9% care in realitate e inutila. Iata cum am abordat problema:

1. **Zero Data Leakage (Split Temporal strict):** Nu am folosit un split random. Am sortat tranzactiile cronologic si am taiat setul de date in Train (70%), Validation (15%) si Test (15%). Astfel, modelul nu a "vazut" informatii din viitor la antrenare.
2. **Optimizare pentru Business (Threshold Tuning):** La un prag standard de 0.5, modelul prindea fraudele, dar bloca eronat 351 de tranzactii legitime (False Positives). Am calibrat pragul de decizie la **0.80**, reducand alarmele false la doar **10**, sacrificand doar un procent infim din fraudele detectate.
3. **Metrici corecte:** Am evaluat performanta folosind **PR-AUC**, **F1-Score** si matricea de confuzie, ignorand complet acuratetea.
4. **Experiment Tracking:** Orice rulare a modelului si orice hiperparametru modificat (ex: `learning_rate`, `max_depth`) sunt salvate automat si versionate in MLflow.

---

## 📂 Structura si Dataset

Datasetul folosit este [Credit Card Fraud Detection de pe Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfile). 
*Nota: Fisierele `.csv` masive, impreuna cu `mlruns/` si baza de date `mlflow.db`, sunt excluse din repo via `.gitignore` pentru a pastra istoricul curat.*

```text
fraud-detection-mlops/
├── api/
│   └── main.py          # Microserviciul FastAPI
├── data/
│   ├── processed/       # Train/Val/Test split (generate local)
│   └── raw/             # Aici pui creditcard.csv descarcat manual
├── src/
│   ├── split.py         # Logica de impartire cronologica
│   ├── train.py         # Antrenare XGBoost & MLflow logging
│   └── evaluate.py      # Analiza pragurilor de decizie
├── .gitignore
├── requirements.txt
└── README.md
