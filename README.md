# MLOps Credit Card Fraud Detection Pipeline

Un pipeline complet de MLOps pentru detectarea fraudelor cu carduri de credit, construit ca proiect de portofoliu. Proiectul acopera intregul flux: de la split-ul temporal al datelor (pentru a evita data leakage-ul), antrenarea unui model XGBoost cu hiperparametri optimizati, logarea experimentelor in MLflow, pana la servirea modelului printr-un API de productie cu FastAPI.

---

## 1. Structura Proiectului & Dataset

Datasetul folosit este cel de pe Kaggle (`Credit Card Fraud Detection`). Deoarece datele sunt voluminoase si sensibile, acestea, alaturi de baza de date MLflow, sunt excluse din controlul versiunilor prin `.gitignore`.

* **Date brute:** Descarca `creditcard.csv` de pe Kaggle si plaseaza-l in folderul `data/raw/`.
* **Organizare cod:**
  * `src/split.py` - Realizeaza split-ul cronologic strict.
  * `src/train.py` - Antreneaza modelul XGBoost si logheaza totul in MLflow.
  * `src/evaluate.py` - Analizeaza si optimizeaza pragul de decizie (threshold tuning).
  * `api/main.py` - Microserviciul FastAPI pentru inferenta in timp real.

---

## 2. Arhitectura & Decizii de Proiectare

Pentru a evita capcanele clasice din machine learning, am implementat urmatoarele reguli:
* **Split Temporal Strict:** Pentru ca datele au o componenta temporala (`Time`), am sortat tranzacțiile cronologic inainte de impartire (70% train, 15% validare, 15% test). Astfel, modelul nu "vede" niciodata viitorul in momentul antrenarii.
* **Metrici pentru Clase Dezechilibrate:** Evitam acuratectea (care e inselatoare cand 99.8% din tranzactii sunt sigure) si ne concentram pe **PR-AUC**, **Precision**, **Recall** si **Matricea de Confuzie**.
* **Optimizarea Pragului (Threshold Tuning):** Folosirea pragului implicit de 0.5 genera prea multe alarme false (351 FP pe validare). Testand mai multe praguri, am stabilit ca o valoare de **0.80 / 0.92** ofera cel mai bun echilibru, reducand drastic alarmele false fara sa ratam fraudele majore.
* **Protectia Setului de Test:** Setul de test a fost utilizat o singura data, la final, pentru evaluarea obiectiva a performantei.

---

## 3. Ghid de Rulare (Quickstart)

### Rulare Locala
1. Activeaza mediul virtual si instaleaza dependentele:
   ```bash
   python -m venv venv
   # Pe Windows:
   venv\Scripts\activate
   pip install -r requirements.txt
