import argparse
import json
from pathlib import Path


def run(dt_dir, lr_dir, rf_dir, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def load_metrics(path):
        with open(Path(path) / "metrics.json") as f:
            return json.load(f)

    results = {
        "DecisionTree": load_metrics(dt_dir),
        "LogisticRegression": load_metrics(lr_dir),
        "RandomForest": load_metrics(rf_dir),
    }

    with open(output_dir / "all_metrics.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dt_dir", type=str)
    p.add_argument("--lr_dir", type=str)
    p.add_argument("--rf_dir", type=str)
    p.add_argument("--output_dir", type=str)
    args = p.parse_args()
    run(args.dt_dir, args.lr_dir, args.rf_dir, args.output_dir)