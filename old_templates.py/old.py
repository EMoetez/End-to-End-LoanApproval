# # API for prediction
# @app.post("/predict/")
# async def make_prediction(data: dict): #, db: Session = Depends(get_db) to add when db is configured
#     try:
#         # Log the input
#         #logging.info(f"Input data: {data}")
        
#         # convert dict to dataframe 
#         input_data = pd.DataFrame([data])
        
#         # Save the input data as a Parquet file
#         #parquet_path = "/temp_data/input_data.parquet"
#         input_data.to_parquet(working_raw_directory, engine='pyarrow', index=False)
        
#         # preprocess data
#         data_processed= preprocess_data(working_raw_directory,working_processed_directory)
        
#         # load the model
#         model= load_model()
        
#         # # Load the test data
#         # test_data_loaded = spark.read.parquet(sample_processed_directory)
        
#         # Make the prediction
#         rf_predictions = predict(working_processed_directory, model)
        
#         # # Log the result to the database
#         # log = PredictionLog(input_data=str(data), prediction=str(rf_predictions["prediction"]))
#         # db.add(log)
#         # db.commit()
#         # db.refresh(log)

#         return rf_predictions
    
#     except Exception as e:
#         logging.error(f"Prediction error: {str(e)}")
#         raise HTTPException(status_code=500, detail="An error occurred during prediction")

# def gradio_predict_from_json(file_path):
#     """
#     Gradio function to interact with FastAPI's predict logic.
#     """
#     try:
#         with open(file_path, "r") as f:
#             data = json.load(f)

#         # Convert dict to DataFrame
#         input_data = pd.DataFrame([data])
        
#         # Save the input data as a Parquet file
#         input_data.to_parquet(working_raw_directory, engine='pyarrow', index=False)
        
#         # Preprocess data
#         data_processed = preprocess_data(working_raw_directory, working_processed_directory)
        
#         # Make the prediction
#         rf_predictions = predict(working_processed_directory, rf_model)
        
#         return "Approved" if (rf_predictions["prediction"] == 1) else "Rejected"
            
#     except Exception as e:
#         return f"error in json {str(e)}"

# # URL of the FastAPI prediction endpoint


# # Create Gradio interface
# gradio_app = gr.Interface(
#     fn=gradio_predict_from_json,
#     inputs=gr.File(label="Upload a JSON File", type="filepath"),
#     outputs="text",
#     title="Prediction Interface",
#     description="Upload a JSON file to get predictions."
# )

# # Serve Gradio at `/predict/`
# @app.get("/predict/", response_class=HTMLResponse)
# async def serve_gradio():
#     """
#     Serve the Gradio app when accessing `/predict/`.
#     """
#     return gradio_app.launch(share=False, inline=True)

# #spark.stop()
