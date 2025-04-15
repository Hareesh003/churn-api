import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from mlflow.models.signature import infer_signature

DataPath= "data/processed/cleaned.csv"
Model_output="models/random_forest.pkl"

def main():
    df=pd.read_csv("data\processed\cleaned.csv")

    X=df.drop(columns=["Churn"])
    X=X.astype("float64")
    y=df["Churn"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    rf=RandomForestClassifier(n_estimators=100,random_state=10)
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)

    acc= accuracy_score(y_test,y_pred)
    f1= f1_score(y_test,y_pred)

    signature = infer_signature(X_train,rf.predict(X_train))

    input_example =X_train.sample(1)

    with mlflow.start_run(run_name="RandomForest-DVC"):
        mlflow.log_param("n_estimators",100)
        mlflow.log_param("random_state",10)
        mlflow.log_metric("accuracy",acc)
        mlflow.log_metric("f1_score",f1)
        mlflow.sklearn.log_model(
            rf,
            artifact_path="random-forest-model",
            signature=signature,
            input_example=input_example
        )
    
    os.makedirs(os.path.dirname(Model_output),exist_ok=True)
    import joblib
    joblib.dump((rf, list(X.columns)), Model_output)
    print(f"✅ Model saved to: {Model_output}")
    print(f"✅ Metrics: accuracy={acc:.4f}, f1={f1:.4f}")

if __name__ == "__main__":
    main()