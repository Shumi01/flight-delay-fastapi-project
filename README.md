# ✈️ Flight Delay Prediction API

This project uses an LSTM model trained with TensorFlow to predict the probability of flight delays. It's served using FastAPI and containerized with Docker.

## 🔍 What It Does

- Predicts the likelihood of a flight being delayed using historical features
- Provides an API endpoint for real-time prediction
- Uses a trained model and scaler

## 📁 Files

- `app.py`: FastAPI app for predictions
- `flight_delay_model.h5`: Trained LSTM model
- `scaler.pkl`: Feature scaler
- `Dockerfile`: Instructions to build a Docker container
- `requirements.txt`: Dependencies for the app

## 🚀 Running the App (Docker)

```bash
docker build -t flight-delay-api .
docker run -p 8000:8000 flight-delay-api
