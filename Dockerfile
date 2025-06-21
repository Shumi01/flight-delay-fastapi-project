FROM tensorflow/tensorflow:2.10.0

# Set working directory in the container
WORKDIR /app

# Copy model, scaler, and app script
COPY flight_delay_model.h5 ./flight_delay_model.h5
COPY scaler.pkl ./scaler.pkl
COPY app.py ./app.py
COPY requirements.txt ./requirements.txt

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
