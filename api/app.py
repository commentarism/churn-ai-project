from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# load model
with open('../model/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(age: int, purchases: int):
    data = np.array([[age, purchases]])
    prediction = model.predict(data)[0]
    return {"churn": int(prediction)}