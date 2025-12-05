import argparse
import json
import joblib
from pathlib import Path
from azureml.core import Workspace, Model
import mlflow
import mlflow.sklearn


def run(dt_metrics_path, lr_metrics_path, rf_metrics_path,
        dt_model_dir, lr_model_dir, rf_model_dir, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # Load metrics
    # -------------------------------
    def load_metrics(uri_folder_path):
        folder = Path(uri_folder_path)
        # Assume only 1 file in the folder, take the first
        file_list = list(folder.glob("*.json"))
        if not file_list:
            raise FileNotFoundError(f"No JSON file found in {folder}")
        metrics_file = file_list[0]
        with open(metrics_file, "r") as f:
            return json.load(f)

    dt_metrics = load_metrics(dt_metrics_path)
    lr_metrics = load_metrics(lr_metrics_path)
    rf_metrics = load_metrics(rf_metrics_path)

    model_results = {
        "DecisionTree": {"metrics": dt_metrics, "model_dir": Path(dt_model_dir)},
        "LogisticRegression": {"metrics": lr_metrics, "model_dir": Path(lr_model_dir)},
        "RandomForest": {"metrics": rf_metrics, "model_dir": Path(rf_model_dir)},
    }

    # -------------------------------
    # Determine best model by accuracy
    # -------------------------------
    best_model_name = None
    best_accuracy = -1
    for model_name, info in model_results.items():
        acc = info["metrics"]["accuracy"]
        print(f"[INFO] {model_name} accuracy = {acc}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = model_name

    print("======================================")
    print(f"Best model: {best_model_name}")
    print(f"Accuracy: {best_accuracy}")
    print("======================================")

    best_model_path = model_results[best_model_name]["model_dir"] / "model.pkl"
    best_model_obj = joblib.load(best_model_path)

    # -------------------------------
    # Azure ML registration
    # -------------------------------
    #ws = Workspace(
    #subscription_id="21a87bf5-6e2d-4b25-9722-6537add69371",
    #resource_group="elliott-m14_CW2",
    #workspace_name="CW2-COM774_elliott")
    
    #registered_model = Model.register(
        #workspace=ws,
        #model_path=str(best_model_path),
        #model_name="BestAccuracyModel",
        #tags={"accuracy": str(best_accuracy)},
        #description=f"Best performing model based on accuracy ({best_model_name})",
    #)
    #print(f"[INFO] Registered model '{registered_model.name}' version {registered_model.version}")

    # -------------------------------
    # Save summary
    # -------------------------------
    summary = {"best_model": best_model_name, "best_accuracy": float(best_accuracy)}
    summary_path = output_dir / "best_model_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"[INFO] Saved summary → {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # Inputs: metrics JSON files
    p.add_argument("--dt_metrics", type=str)
    p.add_argument("--lr_metrics", type=str)
    p.add_argument("--rf_metrics", type=str)

    # Inputs: trained model dirs
    p.add_argument("--dt_model_dir", type=str)
    p.add_argument("--lr_model_dir", type=str)
    p.add_argument("--rf_model_dir", type=str)

    # Output folder
    p.add_argument("--output_dir", type=str)

    args = p.parse_args()

    run(
        args.dt_metrics,
        args.lr_metrics,
        args.rf_metrics,
        args.dt_model_dir,
        args.lr_model_dir,
        args.rf_model_dir,
        args.output_dir
    )