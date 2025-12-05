import argparse
import json
from pathlib import Path


def run(dt_dir, lr_dir, rf_dir, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def load_metrics(metrics_dir):
        metrics_path = Path(metrics_dir) / "metrics.json"
        with open(metrics_path, "r") as f:
            return json.load(f)

    # Collect metrics from all three model training outputs
    results = {
        "DecisionTree": load_metrics(dt_dir),
        "LogisticRegression": load_metrics(lr_dir),
        "RandomForest": load_metrics(rf_dir),
    }

    # Save combined results
    with open(output_dir / "all_metrics.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dt_dir", type=str, required=True)
    parser.add_argument("--lr_dir", type=str, required=True)
    parser.add_argument("--rf_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    run(
        args.dt_dir,
        args.lr_dir,
        args.rf_dir,
        args.output_dir
    )