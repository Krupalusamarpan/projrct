import os
import joblib
from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    model = None
    scaler = None
    print("Model/scaler not found. Train first with python src/train.py")

@app.get("/")
def home():
    return {"message":"Predictive Maintenance API"}

class PredictionInput(BaseModel):
    data: list[float]

@app.post("/predict")
def predict(input_data: PredictionInput):
    if model is None or scaler is None:
        return {"error": "Model/scaler not trained yet. Run python src/train.py first."}
    values = np.array(input_data.data).reshape(1, -1)
    values_scaled = scaler.transform(values)
    prediction = model.predict(values_scaled)
    return {"Predicted_RUL": float(prediction[0])}

