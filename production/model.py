import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import argparse
import joblib

import mlflow
import mlflow.sklearn

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# -----------------------------
# Fairness Metric Functions
# -----------------------------
def demographic_parity_difference(y_pred, protected_attribute):
    groups = protected_attribute.unique()
    if len(groups) != 2:
        return np.nan
    g0, g1 = groups
    rate_0 = y_pred[protected_attribute == g0].mean()
    rate_1 = y_pred[protected_attribute == g1].mean()
    return float(rate_1 - rate_0)


def equalized_odds_difference(y_true, y_pred, protected_attribute):
    groups = protected_attribute.unique()
    if len(groups) != 2:
        return np.nan
    g0, g1 = groups

    def tpr(y_t, y_p):
        cm = confusion_matrix(y_t, y_p)
        if cm.shape != (2, 2):
            return np.nan
        tn, fp, fn, tp = cm.ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    tpr_0 = tpr(y_true[protected_attribute == g0], y_pred[protected_attribute == g0])
    tpr_1 = tpr(y_true[protected_attribute == g1], y_pred[protected_attribute == g1])
    return float(tpr_1 - tpr_0)


# -----------------------------
# Only run code if script is executed directly
# -----------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--trainingdata", type=str, required=True)
    parser.add_argument("--testingdata", type=str, required=True)
    args = parser.parse_args()

    train = pd.read_csv(args.trainingdata)
    test = pd.read_csv(args.testingdata)

    X_train = train.drop(columns=['final_result'])
    y_train = train['final_result']

    X_test = test.drop(columns=['final_result'])
    y_test = test['final_result']

    protected_attribute_test = test["gender"]

    mlflow.autolog(log_input_examples=True)

    models = {
        "DecisionTree": DecisionTreeClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=200),
        "RandomForest": RandomForestClassifier(n_estimators=100)
    }

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)

            dpd = demographic_parity_difference(y_pred, protected_attribute_test)
            eod = equalized_odds_difference(y_test, y_pred, protected_attribute_test)

            mlflow.log_metric("demographic_parity_difference", dpd)
            mlflow.log_metric("equalized_odds_difference", eod)

            feature_names = np.array(X_train.columns)
            importances = None
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_[0])

            if importances is not None:
                top_idx = np.argsort(importances)[-3:][::-1]
                for rank, idx in enumerate(top_idx, 1):
                    feature_name = feature_names[idx]
                    feature_value = float(importances[idx])
                    mlflow.log_metric(f"top_feature_{rank}_importance", feature_value)
                    mlflow.log_metric(f"top_feature_{rank}_name", idx)
                    mlflow.log_param(f"top_feature_{rank}_name_readable", feature_name)
                    print(f"Top {rank}: {feature_name} ({feature_value:.4f})")

            mlflow.sklearn.log_model(sk_model=model, artifact_path=model_name)
            print(f"\nLogged model: {model_name}")
            print(f"Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}")
            print(f"DPD={dpd:.4f}, EOD={eod:.4f}")
            print("Top features logged.\n")