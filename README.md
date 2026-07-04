# FraudShield

Credit Card Fraud Detection System using Machine Learning and MLOps.

## Features

- Data Ingestion
- Data Preprocessing
- Random Forest
- XGBoost
- Cross Validation
- GridSearchCV
- MLflow Tracking
- FastAPI Prediction API
- Docker Containerization
- GitHub Actions CI

## Project Structure

FraudShield/

├── api/

├── data/

├── models/

├── src/

│ ├── components/

│ ├── mlops/

│ └── utils/

├── .github/

├── Dockerfile

├── requirements.txt

└── main.py

## Train Model

python main.py

## Run MLflow

mlflow ui

Open:

http://127.0.0.1:5000

## Run API

uvicorn api.app:app --reload

Swagger:

http://127.0.0.1:8000/docs

## Docker

docker build -t fraudshield .

docker run -p 8000:8000 fraudshield

## Dataset

Download the Credit Card Fraud Detection dataset from Kaggle and place it in:

data/raw/creditcard.csv