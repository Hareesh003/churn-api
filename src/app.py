from fastapi import FastAPI, Body
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model and feature order.
# Ensure your training code saved the model as a tuple: (model, list(X.columns))
model, feature_order = joblib.load("models/random_forest.pkl")
print("Loaded feature order:", feature_order)

# Initialize FastAPI app.
app = FastAPI(title="Churn Prediction API", version="0.1.0")

# Define the input schema matching the API's expected keys.
class CustomerFeatures(BaseModel):
    gender: float
    SeniorCitizen: float
    Partner: float
    Dependents: float
    tenure: float
    PhoneService: float
    PaperlessBilling: float
    MonthlyCharges: float
    TotalCharges: float
    MultipleLines_No_phone_service: float
    MultipleLines_Yes: float
    InternetService_Fiber_optic: float
    InternetService_No: float
    OnlineSecurity_No_internet_service: float
    OnlineSecurity_Yes: float
    OnlineBackup_No_internet_service: float
    OnlineBackup_Yes: float
    DeviceProtection_No_internet_service: float
    DeviceProtection_Yes: float
    TechSupport_No_internet_service: float
    TechSupport_Yes: float
    StreamingTV_No_internet_service: float
    StreamingTV_Yes: float
    StreamingMovies_No_internet_service: float
    StreamingMovies_Yes: float
    Contract_One_year: float
    Contract_Two_year: float
    PaymentMethod_Credit_card_automatic: float
    PaymentMethod_Electronic_check: float
    PaymentMethod_Mailed_check: float

@app.get("/")
def read_root():
    return {"message": "🚀 Churn Prediction API is running!"}

@app.post("/predict")
def predict(data: CustomerFeatures = Body(...)):
    try:
        # Define explicit mappings for the keys that need renaming.
        mapping = {
            "MultipleLines_No_phone_service": "MultipleLines_No phone service",
            "InternetService_Fiber_optic": "InternetService_Fiber optic",
            "Contract_One_year": "Contract_One year",
            "Contract_Two_year": "Contract_Two year",
            "PaymentMethod_Credit_card_automatic": "PaymentMethod_Credit card (automatic)",
            "PaymentMethod_Electronic_check": "PaymentMethod_Electronic check",
            "PaymentMethod_Mailed_check": "PaymentMethod_Mailed check",
            "OnlineSecurity_No_internet_service": "OnlineSecurity_No internet service",
            "OnlineBackup_No_internet_service": "OnlineBackup_No internet service",
            "DeviceProtection_No_internet_service": "DeviceProtection_No internet service",
            "TechSupport_No_internet_service": "TechSupport_No internet service",
            "StreamingTV_No_internet_service": "StreamingTV_No internet service",
            "StreamingMovies_No_internet_service": "StreamingMovies_No internet service"
        }
        
        input_dict_raw = data.dict()
        input_dict_corrected = {}
        
        # Process each input field using the mapping dictionary.
        for k, v in input_dict_raw.items():
            corrected_key = mapping.get(k, k)
            input_dict_corrected[corrected_key] = v

        # Debug: Print corrected inputs and expected features.
        print("Corrected input dictionary:", input_dict_corrected)
        print("Expected feature order:", feature_order)
        for col in feature_order:
            if col not in input_dict_corrected:
                print("WARNING: Expected feature not found in input:", repr(col))
        
        # Build the input array in the precise order expected by the model.
        input_array = np.array([[input_dict_corrected[col] for col in feature_order]])
    
    except KeyError as e:
        return {"error": f"Missing feature in request: {str(e)}"}
    
    # Generate prediction.
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]
    
    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4)
    }
