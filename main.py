# main.py
from fastapi import FastAPI, HTTPException, Depends, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from Model_Deployment.model.load_predict import load_model, predict, preprocess_data
import logging
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql import SparkSession
import pandas as pd
from pathlib import Path
import json



# Define paths
path="Model_deployment/model"
sample_processed_directory = "Model Development/sample/processed"
sample_raw_directory = "Model Development/sample/raw"
working_processed_directory = "Model_Deployment/working/processed"
working_raw_directory = "Model_Deployment/working/raw"



# Start a spark session
spark = SparkSession.builder \
    .appName("CreditCardApprovalPrediction") \
    .config("spark.ui.port", "4040") \
    .getOrCreate() 
    
# Initialize FastAPI
app = FastAPI()


# Load the model once when the app starts
templates = Jinja2Templates(directory="templates")

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

try:
    rf_model = load_model()
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Model file not found")

#####################################################################
@app.get("/predict/", response_class=HTMLResponse)
async def serve_upload_form(request: Request):
    """
    Serve the HTML form for uploading a JSON file.
    """
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/predict/", response_class=HTMLResponse)
async def handle_file_upload(request: Request, file: UploadFile = File(...)):
    """
    Handle the file upload and return the prediction.
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        data = json.loads(contents)
        print(data)
        # Convert dict to DataFrame
        input_data = pd.DataFrame([data])
        
        # Save the input data as a Parquet file
        input_data.to_parquet(working_raw_directory, engine='pyarrow', index=False)
        
        # Preprocess data
        data_processed = preprocess_data(working_raw_directory, working_processed_directory)
        
        # Make the prediction
        rf_predictions = predict(working_processed_directory, rf_model)
        
        prediction = "Approved" if (rf_predictions["prediction"] == 1) else "Rejected"
        return templates.TemplateResponse("upload.html", {"request": request, "prediction": prediction})
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")


#####################################################################

# Start FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
