import unittest
import tempfile
import os
import pandas as pd
import mlflow
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import demographic_parity_difference, equalized_odds_difference

class TestModelIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load small example dataset from your test folder
        cls.train = pd.read_csv("production/tests/data/oula_train_pt1.csv")
        cls.test = pd.read_csv("production/tests/data/oula_test.csv")
        cls.X_train = cls.train.drop(columns=['final_result'])
        cls.y_train = cls.train['final_result']
        cls.X_test = cls.test.drop(columns=['final_result'])
        cls.y_test = cls.test['final_result']
        cls.protected_attr = cls.test['gender']

    def _run_model_checks(self, model, tmpdir):
        # Train model
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        # Basic checks
        self.assertEqual(len(y_pred), len(self.y_test))
        self.assertTrue(all([p in self.y_test.unique() for p in y_pred]))

        # Accuracy check
        acc = (y_pred == self.y_test).mean()
        self.assertGreater(acc, 0)
        self.assertLessEqual(acc, 1)

        # Fairness metrics
        dpd = demographic_parity_difference(y_pred, self.protected_attr)
        eod = equalized_odds_difference(self.y_test, y_pred, self.protected_attr)
        self.assertIsInstance(dpd, float)
        self.assertIsInstance(eod, float)

        # Save model
        model_path = os.path.join(tmpdir, type(model).__name__)
        mlflow.sklearn.save_model(model, model_path)
        self.assertTrue(os.path.exists(model_path))

        # Load model
        loaded_model = mlflow.sklearn.load_model(model_path)
        self.assertEqual(type(model), type(loaded_model))
        # For tree or logistic, check the number of features matches
        if hasattr(model, "coef_"):
            self.assertEqual(model.coef_.shape[1], loaded_model.coef_.shape[1])
        elif hasattr(model, "feature_importances_"):
            self.assertEqual(len(model.feature_importances_), len(loaded_model.feature_importances_))

    def test_decision_tree_integration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._run_model_checks(DecisionTreeClassifier(), tmpdir)

    def test_logistic_regression_integration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._run_model_checks(LogisticRegression(max_iter=200), tmpdir)

    def test_random_forest_integration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._run_model_checks(RandomForestClassifier(n_estimators=100), tmpdir)


if __name__ == "__main__":
    unittest.main()