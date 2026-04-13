from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI
app = FastAPI(
    title="Wine Quality Prediction API",
    description="Inference API for predicting wine quality using trained Random Forest ML model",
    version="1.0"
)

# Load trained model
model = joblib.load("outputs/models/model.pkl")


# -----------------------------
# Request Schema (JSON input)
# -----------------------------
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "Wine Quality Prediction API is running"}


# -----------------------------
# Health Check (required for Jenkins)
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict_wine_quality(data: WineFeatures):

    features = np.array([[ 
        data.fixed_acidity,
        data.volatile_acidity,
        data.citric_acid,
        data.residual_sugar,
        data.chlorides,
        data.free_sulfur_dioxide,
        data.total_sulfur_dioxide,
        data.density,
        data.pH,
        data.sulphates,
        data.alcohol
    ]])

    prediction = model.predict(features)

    return {
        "name": "Sandeep Kumar",
        "roll_no": "2022BCS0076",
        "prediction": int(prediction[0])
    }