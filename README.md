# End-to-End-LoanApproval

This project is an end-to-end solution for predicting loan approval using FastAPI and a machine learning model. The application allows users to upload a JSON file containing loan application data and returns a prediction of whether the loan will be approved or rejected.


## Features

- **FastAPI**: A modern, fast (high-performance), web framework for building APIs with Python 3.6+.
- **Machine Learning**: Uses a RandomForestClassificationModel for predicting loan approval.
- **Spark**: Utilizes Apache Spark for data processing.
- **HTML Interface**: Provides a simple HTML interface for uploading JSON files and displaying predictions.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-username/End-to-End-LoanApproval.git
    cd End-to-End-LoanApproval
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Start the FastAPI application**:
    ```sh
    python main.py
    ```

## Usage

1. **Navigate to the Upload Page**:
    Open your web browser and go to `http://127.0.0.1:8000/predict/`.

2. **Upload a JSON File**:
    Upload a JSON file containing the loan application data. The JSON file should have the following structure:
    ```json
    {
        "feature1": 1.0,
        "feature2": 2.0,
        "feature3": 3.0,
        "feature4": 4.0
    }
    ```
    **You can check features types in "Model Develoment/testV1.ipynb"**

3. **View the Prediction**:
    After uploading the file, the application will display the prediction result: "Approved" or "Rejected".

## File Descriptions

- **main.py**: The main FastAPI application file that handles file uploads, data processing, and predictions.
- **Model_Deployment/model/load_predict.py**: Contains functions for loading the model, preprocessing data, and making predictions.
- **templates/upload.html**: The HTML template for the file upload interface.
- **static/background.jpg**: The background image used in the HTML template.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [Apache Spark](https://spark.apache.org/)
- [Pandas](https://pandas.pydata.org/)
- [Jinja2](https://palletsprojects.com/p/jinja/)
