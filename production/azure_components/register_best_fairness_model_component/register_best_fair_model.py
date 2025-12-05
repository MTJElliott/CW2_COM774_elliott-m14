import argparse
import json
import joblib
from pathlib import Path
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
        file_list = list(folder.glob("*.json"))
        if not file_list:
            raise FileNotFoundError(f"No JSON file found in {folder}")
        metrics_file = file_list[0]
        with open(metrics_file, "r") as f:
            return json.load(f)

    dt_metrics = load_metrics(dt_metrics_path)
    lr_metrics = load_metrics(lr_metrics_path)
    rf_metrics = load_metrics(rf_metrics_path)

    # Bundle for iteration
    model_results = {
        "DecisionTree": {
            "metrics": dt_metrics,
            "model_dir": Path(dt_model_dir),
        },
        "LogisticRegression": {
            "metrics": lr_metrics,
            "model_dir": Path(lr_model_dir),
        },
        "RandomForest": {
            "metrics": rf_metrics,
            "model_dir": Path(rf_model_dir),
        },
    }

    # ----------------------------------------------------
    # Determine best model by fairness (min |dpd|)
    # ----------------------------------------------------
    best_fairness_model_name = None
    best_dpd_abs = float("inf")

    for model_name, info in model_results.items():
        print(f"{model_name}: {info}")
        dpd = info["metrics"].get("demographic_parity_difference", None)

        if dpd is None:
            continue

        print(f"[INFO] {model_name} dpd = {dpd}")

        if abs(dpd) < best_dpd_abs:
            best_dpd_abs = abs(dpd)
            best_fairness_model_name = model_name

    if best_fairness_model_name is None:
        raise ValueError("No valid DPD found in metrics files.")

    print("======================================")
    print(f"Best fairness model: {best_fairness_model_name}")
    print(f"Min |DPD|: {best_dpd_abs}")
    print("======================================")

    # ----------------------------------------------------
    # Load best fairness model
    # ----------------------------------------------------
    best_model_path = model_results[best_fairness_model_name]["model_dir"] / "model.pkl"
    best_model_obj = joblib.load(best_model_path)

    # -------------------------------
    # Save summary
    # -------------------------------
    summary = {
        "best_model": best_fairness_model_name,
        "best_dpd_abs": float(best_dpd_abs)
    }

    summary_path = output_dir / "best_model_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"[INFO] Saved summary → {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # Inputs: metrics JSON files
    p.add_argument("--dt_metrics", type=str, required=True)
    p.add_argument("--lr_metrics", type=str, required=True)
    p.add_argument("--rf_metrics", type=str, required=True)

    # Inputs: trained model output dirs
    p.add_argument("--dt_model_dir", type=str, required=True)
    p.add_argument("--lr_model_dir", type=str, required=True)
    p.add_argument("--rf_model_dir", type=str, required=True)

    # Output folder
    p.add_argument("--output_dir", type=str, required=True)

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