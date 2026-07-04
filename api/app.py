from fastapi import FastAPI
from pydantic import BaseModel         #pydantic used for data validation
import numpy as np
import joblib

app = FastAPI()

#loads the model
model = joblib.load("models/best_random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")             # loads the scaler from saved or dumped

class Transaction(BaseModel):
    features: list[float]      # this accepts only list of float values

@app.get("/")
def home():
    return {
        "message": "---FRAUDSHIELD API WORKING---"
    }

@app.post("/predict")
def predict(data: Transaction):

    arr = np.array(data.features).reshape(1, -1)

    # scale the data form the inputs -- scale used reduce the range of the values in the columns
    arr = scaler.transform(arr)
    prediction = model.predict(arr)[0]
    probability = model.predict_proba(arr)[0][1]

    return {
        "prediction": int(prediction),
        "fraud_probability": float(probability)
    }