import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.tracking import load_serving_model, load_threshold

model, MODEL_RUN_ID = load_serving_model()
THRESHOLD = load_threshold(MODEL_RUN_ID)
print(f"Model incarcat din run-ul {MODEL_RUN_ID}, prag aplicat: {THRESHOLD:.4f}")

if hasattr(model, "feature_names_in_"):
    FEATURES = list(model.feature_names_in_)
else:
    FEATURES = list(model.get_booster().feature_names)

EXAMPLE_FEATURES = {name: 0.0 for name in FEATURES}

app = FastAPI(title="Fraud Detection API", version="1.0")


class Transaction(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description=f"Cele {len(FEATURES)} feature-uri: {', '.join(FEATURES)}",
        examples=[EXAMPLE_FEATURES],
    )


class Prediction(BaseModel):
    is_fraud: bool
    fraud_probability: float
    threshold_applied: float


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_run_id": MODEL_RUN_ID,
        "threshold": THRESHOLD,
        "n_features": len(FEATURES),
    }


@app.post(
    "/predict",
    response_model=Prediction,
    responses={422: {"description": "Lipsesc feature-uri sau sunt in plus"}},
)
def predict_fraud(transaction: Transaction):
    given = set(transaction.features)
    expected = set(FEATURES)
    missing = sorted(expected - given)
    extra = sorted(given - expected)
    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "unexpected_features": extra},
        )

    try:
        df = pd.DataFrame([transaction.features])[FEATURES]
        prob_fraud = float(model.predict_proba(df)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eroare la predictie: {e}")

    return Prediction(
        is_fraud=prob_fraud >= THRESHOLD,
        fraud_probability=round(prob_fraud, 4),
        threshold_applied=THRESHOLD,
    )
