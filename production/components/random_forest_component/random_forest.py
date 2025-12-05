import argparse
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def demographic_parity_difference(y_pred, protected_attribute):
    groups = protected_attribute.unique()
    if len(groups) != 2:
        return None
    g0, g1 = groups
    return float(y_pred[protected_attribute == g1].mean()
                 - y_pred[protected_attribute == g0].mean())


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
    return float(tpr_1 - tpr_0)


def run(train_path, test_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train = train.drop(columns=["final_result"])
    y_train = train["final_result"]

    X_test = test.drop(columns=["final_result"])
    y_test = test["final_result"]
    protected = test["gender"]

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "model_name": "RandomForest",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "dpd": demographic_parity_difference(y_pred, protected),
        "eod": equalized_odds_difference(y_test, y_pred, protected),
    }

    # Save model
    joblib.dump(model, output_dir / "model.pkl")

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trainingdata", type=str)
    p.add_argument("--testingdata", type=str)
    p.add_argument("--output_dir", type=str)
    args = p.parse_args()
    run(args.trainingdata, args.testingdata, args.output_dir)