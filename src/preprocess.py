import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Use raw strings (r"") to handle backslashes in Windows file paths
RAW_DATA_PATH = r"E:\mlops\data\raw\WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = r"E:\mlops\data\processed\cleaned.csv"

def clean_data(df):
    # Drop rows with empty TotalCharges
    df = df[df["TotalCharges"].str.strip() != ""]

    # 🔧 Typo fixed: "TotalCharfes" → "TotalCharges"
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    # 🔧 Typo fixed: "columsn" → "columns"
    df.drop(columns=["customerID"], inplace=True)

    # Binary encoding
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

    # One-hot encoding
    multi_class_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]
    df = pd.get_dummies(df, columns=multi_class_cols, drop_first=True)

    # Scale numeric columns
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df

def main():
    df = pd.read_csv(RAW_DATA_PATH)
    df_clean = clean_data(df)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"✅ Cleaned data saved to: {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()
