from fastapi import FastAPI
from pydantic import BaseModel         
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("models/best_random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")             

class Transaction(BaseModel):
    features: list[float]      

@app.get("/")
def home():
    return {
        "message": "---FRAUDSHIELD API WORKING---"
    }

@app.post("/predict")
def predict(data: Transaction):
    arr = np.array(data.features).reshape(1, -1)
    arr = scaler.transform(arr)
    prediction = model.predict(arr)[0]
    probability = model.predict_proba(arr)[0][1]

    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability)
    }
