import argparse
import joblib
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier  


def run(train_path, model_output_dir):
    model_output_dir = Path(model_output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    train = pd.read_csv(train_path)

    X_train = train.drop(columns=["final_result"])
    y_train = train["final_result"]

    # Train Decision Tree model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Save trained model to the output folder for Azure ML
    output_path = model_output_dir / "model.pkl"
    joblib.dump(model, output_path)

    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # MUST MATCH component YAML exactly
    parser.add_argument("--trainingdata", type=str, help="Path to training CSV")
    parser.add_argument("--trained_model", type=str, help="Directory to save trained model")

    args = parser.parse_args()

    run(args.trainingdata, args.trained_model)