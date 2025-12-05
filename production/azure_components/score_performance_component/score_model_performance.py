import argparse
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score


def run(model_dir, test_path, output_dir):

    # Prepare output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Locate model
    model_dir = Path(model_dir)
    model_path = model_dir / "model.pkl"

    # ----- Load trained model -----
    model = joblib.load(model_path)

    # Try to infer a friendly model name
    try:
        model_name = type(model).__name__
    except:
        model_name = "UnknownModel"

    # ----- Load test data -----
    test = pd.read_csv(test_path)
    X_test = test.drop(columns=["final_result"])
    y_test = test["final_result"]

    # ----- Predict -----
    y_pred = model.predict(X_test)

    # ----- Compute metrics -----
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall)
    }

    # ----- Write metrics.json for pipeline output -----
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # ----- Log to Azure ML (stdout) -----
    print("===== MODEL PERFORMANCE =====")
    print(f"Model: {model_name}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("=============================")

    # ----- Log metrics using MLflow -----
    #mlflow.autolog(log_models=False, log_input_examples=False, log_model_signatures=False)
    #mlflow.log_param("model_name", model_name)
    #mlflow.log_metric("accuracy", accuracy)
    #mlflow.log_metric("precision", precision)
    #mlflow.log_metric("recall", recall)

    # Log model artifact again for visibility (OPTIONAL)
    #mlflow.sklearn.log_model(model, artifact_path="scored_model")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trained_model", type=str)
    p.add_argument("--testdata", type=str)
    p.add_argument("--output_dir", type=str)
    args = p.parse_args()

    # Enable autolog if desired
    #mlflow.autolog(log_input_examples=False, log_model_signatures=False)
    run(args.trained_model, args.testdata, args.output_dir)
    #with mlflow.start_run():