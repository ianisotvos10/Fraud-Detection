#  MLOps: Credit Card Fraud Detection Pipeline

Obiectivul principal nu a fost doar gasirea unui algoritm cu o acuratete mare, ci construirea unui pipeline end-to-end capabil sa rezolve o problema din lumea reala: detectarea fraudelor bancare in conditii de dezechilibru extrem al datelor. 

Sistemul preia date brute, le proceseaza corect din punct de vedere temporal, antreneaza un model XGBoost, inregistreaza experimentele si expune predictia printr-un API RESTful.

## Arhitectura si Decizii Tehnice

Cand lucrezi cu date tabulare financiare unde fraudele reprezinta sub 0.2% din total, abordarile clasice din tutoriale esueaza in productie. Iata deciziile de inginerie din spatele proiectului:

1. **Prevenirea Data Leakage-ului prin Split Temporal:** 
   O greseala comuna este impartirea aleatoare a datelor (random split). Intr-un mediu bancar real, modelul invata din trecut pentru a prezice viitorul. Daca amestecam datele, modelul ar "vedea" tipare viitoare in timpul antrenamentului, oferind o performanta falsa. Solutia a fost sortarea stricta a tranzactiilor dupa timp (Time) inainte de a le imparti in Train (70%), Validation (15%) si Test (15%).

2. **Metrici relevante pentru dezechilibru extrem:** 
   Daca un model clasifica absolut toate tranzactiile ca fiind "sigure", va avea o acuratete de 99.8%, dar va fi complet inutil. Din acest motiv, evaluarea se face exclusiv prin **PR-AUC (Precision-Recall Curve)**, **F1-Score** si **Matricea de Confuzie**. Ne intereseaza sa prindem fraudele (Recall) fara sa blocam cardurile clientilor legitimi (Precision).

3. **Optimizarea Pragului de Decizie (Threshold Tuning):** 
   Modelul XGBoost returneaza o probabilitate (0-1). Pragul standard de 0.5 bloca fraudele, dar genera sute de alarme false (False Positives). Prin analiza rezultatelor pe setul de validare, am ridicat pragul la **0.80**. Aceasta decizie reduce drastic alarmele false (de la sute la doar cateva zeci), pastrand eficienta sistemului si reducand uzura operationala pentru echipa de suport a bancii.

4. **Trasabilitate si MLflow:** 
   Pentru a nu pierde istoricul antrenarilor, am integrat MLflow. Orice schimbare a hiperparametrilor (learning rate, adancimea arborilor) si metricile rezultate sunt inregistrate automat intr-o baza de date SQLite locala. Modelul final este stocat ca artifact si incarcat direct de acolo.

5. **Servirea Modelului (Inference API):** 
   Un model de Machine Learning trebuie sa fie interogabil. Am impachetat XGBoost-ul intr-un microserviciu scris in FastAPI. Prin intermediul schemelor Pydantic, API-ul valideaza ca datele de intrare sunt corecte (tipuri numerice pentru V1-V28 si Amount) inainte de a le trimite spre predictie, returnand decizia si probabilitatea in timp real.

---

##  Tech Stack
* **Machine Learning:** Python, XGBoost, scikit-learn, Pandas
* **MLOps & Tracking:** MLflow (cu backend SQLite local)
* **Serving & Backend:** FastAPI, Uvicorn, Pydantic
* **Deployment & Rulare:** Docker, Git

---

##  Structura si Dataset

Datasetul folosit este [Credit Card Fraud Detection de pe Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfile). 
*Nota: Fisierele `.csv` masive, impreuna cu `mlruns/` si baza de date `mlflow.db`, sunt excluse din repo via `.gitignore` pentru a pastra istoricul curat.*

```text
fraud-detection-mlops/
├── api/
│   └── main.py          # Microserviciul FastAPI pentru inferenta
├── data/
│   ├── processed/       # Train/Val/Test split (generate local)
│   └── raw/             # Aici pui creditcard.csv descarcat manual
├── src/
│   ├── split.py         # Logica de impartire cronologica
│   ├── train.py         # Antrenare XGBoost & MLflow logging
│   └── evaluate.py      # Optimizarea pragurilor de decizie
├── .gitignore
├── requirements.txt
└── README.md
