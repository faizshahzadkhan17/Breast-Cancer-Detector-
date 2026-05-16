"""
Breast Cancer Detector — FastAPI Backend
=========================================
Serves predictions from the pre-trained ML model via REST endpoints.
"""

import os
import io
import json
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()  # loads .env from project root

# Resolve paths relative to the project root (one level up from /backend)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "src" / "best_model.joblib"
SCALER_PATH = PROJECT_ROOT / "src" / "scaler.joblib"

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# The 30 features in exact order expected by the model
FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]

# ---------------------------------------------------------------------------
# Global state populated at startup
# ---------------------------------------------------------------------------

ml_model = None
ml_scaler = None
supabase_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artifacts and optional Supabase client on startup."""
    global ml_model, ml_scaler, supabase_client

    # Load ML artifacts
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise RuntimeError(f"Scaler file not found: {SCALER_PATH}")

    ml_model = joblib.load(MODEL_PATH)
    ml_scaler = joblib.load(SCALER_PATH)
    print(f"[OK] Model loaded from {MODEL_PATH}")
    print(f"[OK] Scaler loaded from {SCALER_PATH}")

    # Optional Supabase connection
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            from supabase import create_client
            supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
            print("[OK] Supabase client connected")
        except Exception as e:
            print(f"[WARN] Supabase connection failed (predictions won't be saved): {e}")
    else:
        print("[WARN] Supabase credentials not set - history features disabled")

    yield  # Application runs here

    # Cleanup (if needed)
    print("[INFO] Shutting down...")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Breast Cancer Detector API",
    description="Predict breast cancer diagnosis from cell nucleus measurements.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """30 numeric features for a single patient."""
    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="Exactly 30 cell-nucleus feature values.",
    )
    patient_label: str | None = Field(
        default=None,
        description="Optional label / patient identifier.",
    )


class PredictResponse(BaseModel):
    diagnosis: str
    confidence: float
    patient_label: str | None = None


class BatchRow(BaseModel):
    row: int
    diagnosis: str
    confidence: float


class BatchResponse(BaseModel):
    total: int
    malignant: int
    benign: int
    results: list[BatchRow]


class HistoryRecord(BaseModel):
    patient_label: str | None = None
    diagnosis: str
    confidence: float
    input_features: dict | list | None = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _predict(features_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale features and return (predictions, malignancy_probabilities)."""
    X_scaled = ml_scaler.transform(features_2d)
    preds = ml_model.predict(X_scaled)
    probs = ml_model.predict_proba(X_scaled)[:, 1]
    return preds, probs


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check():
    """Simple health / readiness probe."""
    return {"status": "ok", "model_loaded": ml_model is not None}


@app.get("/features")
async def get_feature_names():
    """Return the ordered list of expected feature names."""
    return {"features": FEATURE_NAMES}


@app.post("/predict", response_model=PredictResponse)
async def predict_single(req: PredictRequest):
    """Predict diagnosis for a single set of 30 features."""
    features = np.array(req.features).reshape(1, -1)
    preds, probs = _predict(features)

    diagnosis = "Malignant" if preds[0] == 1 else "Benign"
    confidence = round(float(probs[0]), 4)

    # Auto-save to Supabase if available
    if supabase_client:
        try:
            record = {
                "patient_label": req.patient_label,
                "diagnosis": diagnosis,
                "confidence": confidence,
                "input_features": dict(zip(FEATURE_NAMES, req.features)),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            supabase_client.table("predictions").insert(record).execute()
        except Exception as e:
            print(f"[WARN] Failed to save to Supabase: {e}")

    return PredictResponse(
        diagnosis=diagnosis,
        confidence=confidence,
        patient_label=req.patient_label,
    )


@app.post("/predict-batch", response_model=BatchResponse)
async def predict_batch(file: UploadFile = File(...)):
    """Accept a CSV file (no header, 30 columns) and return predictions for every row."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents), header=None)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    if df.shape[1] != 30:
        raise HTTPException(
            status_code=400,
            detail=f"CSV must have exactly 30 columns. Found {df.shape[1]}.",
        )

    preds, probs = _predict(df.values)

    results = []
    malignant_count = 0
    for i in range(len(preds)):
        diag = "Malignant" if preds[i] == 1 else "Benign"
        if diag == "Malignant":
            malignant_count += 1
        results.append(BatchRow(row=i + 1, diagnosis=diag, confidence=round(float(probs[i]), 4)))

    return BatchResponse(
        total=len(preds),
        malignant=malignant_count,
        benign=len(preds) - malignant_count,
        results=results,
    )


@app.get("/history")
async def get_history():
    """Fetch all prediction records from Supabase."""
    if not supabase_client:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")

    try:
        response = supabase_client.table("predictions").select("*").order("created_at", desc=True).execute()
        return {"records": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {e}")


@app.post("/history")
async def save_history(record: HistoryRecord):
    """Manually save a prediction record to Supabase."""
    if not supabase_client:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")

    try:
        data = {
            "patient_label": record.patient_label,
            "diagnosis": record.diagnosis,
            "confidence": record.confidence,
            "input_features": record.input_features,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        response = supabase_client.table("predictions").insert(data).execute()
        return {"message": "Record saved", "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save record: {e}")


# ---------------------------------------------------------------------------
# Run directly for convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
