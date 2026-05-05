from fastapi import FastAPI
import joblib
import numpy as np

model = joblib.load("../models/model.pkl")

app = FastAPI()

@app.get("/")
def home():

    return {"message":"Predictive Maintenance API"}

@app.post("/predict")

def predict(data:dict):

    values = list(data.values())

    prediction = model.predict([values])

    return {"Predicted_RUL": float(prediction[0])}