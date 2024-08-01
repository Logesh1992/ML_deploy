from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
import numpy as np

# Load the models
knn_model = joblib.load('app/knn_model.joblib')
linear_model = joblib.load('app/linear_model.joblib')

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/')
def read_root():
    return FileResponse('html.html')

@app.post('/predict', response_class=JSONResponse)
def predict(date_str: str = Form(...)):
    # Convert the input date string to a datetime format
    date = pd.to_datetime(date_str, format='%Y-%m-%d')
    
    # Extract year, month, and day for feature array
    year = date.year
    month = date.month
    day = date.day
    
    # Create a feature array for prediction
    features = np.array([[year, month, day]])
    
    # Use KNN model before April 2024 and Linear Regression model after that date
    cutoff_date = pd.to_datetime('2024-04-01')
    if date < cutoff_date:
        predicted_value = knn_model.predict(features)[0]
    else:
        predicted_value = linear_model.predict(features)[0]

    return {'predicted': predicted_value}
