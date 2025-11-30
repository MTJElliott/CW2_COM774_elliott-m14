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

from sklearn.metrics import (accuracy_score, precision_score, recall_score,confusion_matrix)
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference


# -----------------------------
# Argument Parsing
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True)
parser.add_argument("--testingdata", type=str, required=True)
args = parser.parse_args()

train = pd.read_csv(args.trainingdata)
test = pd.read_csv(args.testingdata)

# -----------------------------
# Prepare Data
# -----------------------------
X_train = train.drop(columns=['final_result'])
y_train = train['final_result']

X_test = test.drop(columns=['final_result'])
y_test = test['final_result']

# Protected attribute for fairness metrics
protected_attribute_test = test["gender"]  # <-- change if needed

mlflow.autolog(log_input_examples=True)

models = {
    "DecisionTree": DecisionTreeClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier(n_estimators=100)
}

# -----------------------------
# Train, evaluate, fairness, feature importance
# -----------------------------
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        
        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # ------------------
        # Standard metrics
        # ------------------
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        # ------------------
        # Fairness metrics
        # ------------------
        # Convert to numpy arrays
        y_true_arr = y_test.values
        y_pred_arr = y_pred
        protected_arr = protected_attribute_test.values

        # Demographic Parity Difference
        dpd = demographic_parity_difference(
            y_true=None,      # DP only needs predictions
            y_pred=y_pred_arr,
            sensitive_features=protected_arr
        )

        # Equalized Odds Difference
        eod = equalized_odds_difference(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            sensitive_features=protected_arr
        )

        mlflow.log_metric("demographic_parity_difference", dpd)
        mlflow.log_metric("equalized_odds_difference", eod)

        # ------------------
        # Feature importances
        # ------------------
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            importances = None

        if importances is not None:
            top_idx = np.argsort(importances)[::-1][:3]

            for rank, idx in enumerate(top_idx, 1):
                feature_name = feature_names[idx]
                feature_imp = float(importances[idx])

                mlflow.log_metric(f"top_feature_{rank}_importance", feature_imp)
                mlflow.log_param(f"top_feature_{rank}_name", feature_name)

                print(f"Top {rank}: {feature_name} ({feature_imp:.4f})")

        # ------------------
        # Log model
        # ------------------
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,
            registered_model_name=None
        )

        print(f"\nLogged model: {model_name}")
        print(f"Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}")
        print(f"DPD={dpd:.4f}, EOD={eod:.4f}")
        print("Top features logged.\n")