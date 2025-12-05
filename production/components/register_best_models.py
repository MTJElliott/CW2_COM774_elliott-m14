import argparse
import json
import joblib
from pathlib import Path
import mlflow
import mlflow.sklearn


def run(metrics_file, dt_dir, lr_dir, rf_dir):

    # Load metrics summary
    with open(metrics_file) as f:
        metrics = json.load(f)

    # Select best accuracy model
    best_accuracy_model = None
    best_accuracy = -1

    # Select best fairness model (min abs(dpd))
    best_fairness_model = None
    best_fairness = float("inf")

    for model_name, m in metrics.items():

        if m["accuracy"] > best_accuracy:
            best_accuracy = m["accuracy"]
            best_accuracy_model = model_name

        if abs(m["dpd"]) < best_fairness:
            best_fairness = abs(m["dpd"])
            best_fairness_model = model_name

    print("Best accuracy model:", best_accuracy_model)
    print("Best fairness model:", best_fairness_model)

    # Map model names to folders
    model_dirs = {
        "DecisionTree": dt_dir,
        "LogisticRegression": lr_dir,
        "RandomForest": rf_dir
    }

    # MLflow Registration
    # -----------------------------
    mlflow.set_tracking_uri("azureml://")

    # Register best accuracy model
    with mlflow.start_run(run_name="BestAccuracyModel"):
        model_obj = joblib.load(Path(model_dirs[best_accuracy_model]) / "model.pkl")
        mlflow.sklearn.log_model(
            model_obj,
            artifact_path="model",
            registered_model_name="BestAccuracyModel"
        )
        mlflow.log_metric("best_accuracy", best_accuracy)

    # Register best fairness model
    with mlflow.start_run(run_name="BestFairnessModel"):
        model_obj = joblib.load(Path(model_dirs[best_fairness_model]) / "model.pkl")
        mlflow.sklearn.log_model(
            model_obj,
            artifact_path="model",
            registered_model_name="BestFairnessModel"
        )
        mlflow.log_metric("best_fairness_dpd", best_fairness)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_file", type=str)
    p.add_argument("--dt_dir", type=str)
    p.add_argument("--lr_dir", type=str)
    p.add_argument("--rf_dir", type=str)
    args = p.parse_args()
    run(args.metrics_file, args.dt_dir, args.lr_dir, args.rf_dir)