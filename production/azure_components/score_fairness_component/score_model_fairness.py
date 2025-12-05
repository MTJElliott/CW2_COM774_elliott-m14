import argparse
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix


# ---------------------------
# Fairness metric functions
# ---------------------------

def demographic_parity_difference(y_pred, protected_attribute):
    groups = protected_attribute.unique()
    if len(groups) != 2:
        return None
    g0, g1 = groups
    return float(
        y_pred[protected_attribute == g1].mean()
        - y_pred[protected_attribute == g0].mean()
    )


def equalized_odds_difference(y_true, y_pred, protected_attribute):
    groups = protected_attribute.unique()
    if len(groups) != 2:
        return None

    g0, g1 = groups

    def tpr(y_t, y_p):
        cm = confusion_matrix(y_t, y_p)
        if cm.shape != (2, 2):
            return None
        tn, fp, fn, tp = cm.ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    tpr_0 = tpr(y_true[protected_attribute == g0], y_pred[protected_attribute == g0])
    tpr_1 = tpr(y_true[protected_attribute == g1], y_pred[protected_attribute == g1])

    if tpr_0 is None or tpr_1 is None:
        return None

    return float(tpr_1 - tpr_0)


# ---------------------------
# Main Execution
# ---------------------------

def run(model_dir, test_path, output_dir):

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model directory → locate model.pkl
    model_dir = Path(model_dir)
    model_path = model_dir / "model.pkl"

    # Load model
    model = joblib.load(model_path)

    # Try to infer model name
    try:
        model_name = type(model).__name__
    except:
        model_name = "UnknownModel"

    # Load test data
    test = pd.read_csv(test_path)
    X_test = test.drop(columns=["final_result"])
    y_test = test["final_result"]
    protected = test["gender"]

    # Predictions
    y_pred = model.predict(X_test)

    # Fairness metrics
    dpd = demographic_parity_difference(y_pred, protected)
    eod = equalized_odds_difference(y_test, y_pred, protected)

    metrics = {
        "model_name": model_name,
        "demographic_parity_difference": dpd,
        "equalized_odds_difference": eod
    }

    # Save JSON for pipeline outputs
    with open(output_dir / "fairness_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # STDOUT logs visible in Azure ML job run logs
    print("===== MODEL FAIRNESS =====")
    print(f"Model: {model_name}")
    print(f"Demographic Parity Difference (DPD): {dpd}")
    print(f"Equalized Odds Difference (EOD): {eod}")
    print("==========================")

    # MLflow logging
    #mlflow.autolog(log_models=False, log_input_examples=False, log_model_signatures=False)
    #mlflow.log_param("model_name", model_name)
    #mlflow.log_metric("dpd", dpd)
    #mlflow.log_metric("eod", eod)

    # Optional: log model again for visibility
    #mlflow.sklearn.log_model(model, artifact_path="fairness_model")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trained_model", type=str)
    p.add_argument("--testdata", type=str)
    p.add_argument("--output_dir", type=str)
    args = p.parse_args()

    # MLflow autolog (disabled model signature logging for simplicity)
    #mlflow.autolog(log_input_examples=False, log_model_signatures=False)

    # Wrap everything in a run so metrics appear in Azure UI
    #with mlflow.start_run():
    run(args.trained_model, args.testdata, args.output_dir)