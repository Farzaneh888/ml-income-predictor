from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("model.joblib")

app = FastAPI()

class Person(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    Fare: float

@app.get("/")
def read_root():
    return {"message": "titanic-survival-predictor"}

@app.post("/predict")
def predict(data: Person):
    input_data = np.array([[data.Pclass, data.Sex, data.Age, data.Fare]])
    prediction = model.predict(input_data)[0]
    return {"prediction": int(prediction)}
