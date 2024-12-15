import gradio as gr
import requests
import json

# URL of the FastAPI prediction endpoint
PREDICTION_ENDPOINT = "http://127.0.0.1:8000/predict"

def predict_from_json(file):
    """
    Function to handle predictions using the FastAPI endpoint.
    Accepts a JSON file, reads its content, sends it to the prediction endpoint, 
    and returns the prediction result.
    """
    try:
        # Read the JSON file
        with open(file.name, 'r') as f:
            data = json.load(f)
        
        # Send the data to the FastAPI endpoint
        response = requests.post(PREDICTION_ENDPOINT, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return f"Prediction: {result['prediction']}"
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error processing file: {str(e)}"

# Gradio interface
interface = gr.Interface(
    fn=predict_from_json,
    inputs=gr.File(label="Upload a JSON File", type="filepath"),
    outputs="text",
    title="Prediction Interface",
    description="Upload a JSON file containing the input features to get a prediction."
)




# Launch the Gradio app
if __name__ == "__main__":
    interface.launch()
