from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
import numpy as np

# Load the model
model_fit = joblib.load('app/model.joblib')

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/')
def read_root():
    return FileResponse('html.html')

@app.post('/predict', response_class=JSONResponse)
def predict(date_str: str = Form(...)):
    # Convert the input date string to a datetime format
    date = pd.to_datetime(date_str, format='%Y-%m-%d')
    
    # Convert the date to year number
    year_number = date.year
    
    # Create a feature array for prediction
    features = np.array([[year_number]])
    
    # Forecast the cotton prices for the input date
    predicted_value = model_fit.predict(features)[0]

    return {'predicted': predicted_value}
