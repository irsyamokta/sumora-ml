from fastapi import FastAPI
from app.schemas.input_schema import WaterQualityInput
import joblib
import numpy as np
import os

app = FastAPI(
    title="Water Quality Classification API",
    description="Menentukan apakah air layak minum berdasarkan 4 parameter sensor",
    version="1.0.0"
)

model_path = os.path.join("app", "models", "random_forest_model.pkl")
imputer_path = os.path.join("app", "preprocessing", "imputer.pkl")
scaler_path = os.path.join("app", "preprocessing", "scaler.pkl")

model = joblib.load(model_path)
imputer = joblib.load(imputer_path)
scaler = joblib.load(scaler_path)

@app.get("/")
def read_root():
    return {"message": "Water Quality Classification API is running."}

@app.post("/predict")
def predict_water_quality(data: WaterQualityInput):
    # Ekstrak fitur dari input
    input_data = [[
        data.Sulfate,
        data.ph,
        data.Chloramines,
        data.Solids
    ]]

    transformed = imputer.transform(input_data)
    transformed = scaler.transform(transformed)

    prediction = model.predict(transformed)[0]
    result = "Potable" if prediction == 1 else "Not Potable"

    return {
        "result": result
    }
