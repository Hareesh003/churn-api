# src/evaluate.py

import pandas as pd
import os 
import joblib
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/cleaned.csv"
MODEL_PATH = "models/random_forest.pkl"
CM_PLOT_PATH = "reports/confusion_matrix.png"
ROC_PLOT_PATH = "reports/roc_curve.png"

def main():
    # Load processed dataset
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Load trained model
    model = joblib.load(MODEL_PATH)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # ROC-AUC Score
    auc = roc_auc_score(y_test, y_proba)
    print(f"✅ ROC-AUC: {auc:.4f}")

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    os.makedirs(os.path.dirname(CM_PLOT_PATH), exist_ok=True)
    plt.savefig(CM_PLOT_PATH)
    plt.close()

    # ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    os.makedirs(os.path.dirname(ROC_PLOT_PATH), exist_ok=True)
    plt.savefig(ROC_PLOT_PATH)
    plt.close()

    # MLflow logging
    with mlflow.start_run(run_name="Evaluation Metrics"):
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_artifact(CM_PLOT_PATH)
        mlflow.log_artifact(ROC_PLOT_PATH)
        print("✅ Logged confusion matrix and ROC curve to MLflow")

if __name__ == "__main__":
    main()
