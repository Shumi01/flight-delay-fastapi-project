from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Define input schema
class FlightInput(BaseModel):
    DEP_DELAY: float
    DISTANCE: float
    DAY_OF_WEEK: int

# Load model and scaler
model = load_model("flight_delay_model.h5")
scaler = joblib.load("scaler.pkl")

app = FastAPI()

@app.post("/predict/")
def predict_delay(input: FlightInput):
    try:
        # Convert input to array
        data = np.array([[input.DEP_DELAY, input.DISTANCE, input.DAY_OF_WEEK]])
        
        # Scale
        data_scaled = scaler.transform(data)
        
        # Reshape for LSTM [samples, timesteps, features]
        data_scaled = data_scaled.reshape((1, 1, 3))
        
        # Predict
        prediction = model.predict(data_scaled)
        return {
            "delay_probability": float(prediction[0][0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
