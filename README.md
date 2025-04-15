
# Churn Prediction API

This project implements a **Churn Prediction API** using **FastAPI** for predicting customer churn based on various customer features. The model is a **Random Forest Classifier** trained on the **Telco Customer Churn** dataset.

## Features

- **Predict customer churn**: The API predicts if a customer will churn based on features such as contract type, payment method, service usage, etc.
- **RESTful API**: Built using **FastAPI**.
- **Model**: Trained with **scikit-learn** and exported using **joblib**.
- **Dockerized**: The application is dockerized for easy deployment.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Hareesh003/churn-api.git
    cd churn-api
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    uvicorn src.app:app --host 0.0.0.0 --port 8000
    ```

4. The API will be available at `http://localhost:8000`.

## Docker Setup

To run the application in Docker, build and run the Docker container:

1. Build the Docker image:
    ```bash
    docker build -t churn-api .
    ```

2. Run the Docker container:
    ```bash
    docker run -p 8000:8000 churn-api
    ```

3. The API will be available at `http://localhost:8000`.

## API Endpoints

- **GET /docs**: Swagger UI documentation for the API.
- **POST /predict**: Predict churn probability based on customer data.
   
Example of the request body for the `/predict` endpoint:

```json
{
    "gender": 0,
    "SeniorCitizen": 0,
    "Partner": 0,
    "Dependents": 0,
    "tenure": 0,
    "PhoneService": 0,
    "PaperlessBilling": 0,
    "MonthlyCharges": 0,
    "TotalCharges": 0,
    "MultipleLines_No phone service": 0,
    "MultipleLines_Yes": 0,
    "InternetService_Fiber optic": 0,
    "InternetService_No": 0,
    "OnlineSecurity_No internet service": 0,
    "OnlineSecurity_Yes": 0,
    "OnlineBackup_No internet service": 0,
    "OnlineBackup_Yes": 0,
    "DeviceProtection_No internet service": 0,
    "DeviceProtection_Yes": 0,
    "TechSupport_No internet service": 0,
    "TechSupport_Yes": 0,
    "StreamingTV_No internet service": 0,
    "StreamingTV_Yes": 0,
    "StreamingMovies_No internet service": 0,
    "StreamingMovies_Yes": 0,
    "Contract_One year": 0,
    "Contract_Two year": 0,
    "PaymentMethod_Credit card (automatic)": 0,
    "PaymentMethod_Electronic check": 0,
    "PaymentMethod_Mailed check": 0
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The **Telco Customer Churn** dataset used for training the model is from **Kaggle**.
- **FastAPI** and **Docker** were used for building and containerizing the application.
