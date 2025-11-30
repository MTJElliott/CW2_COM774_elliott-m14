import unittest
import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class TestModelOutput(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Get the directory this test file is in
        test_dir = os.path.dirname(__file__)
        cls.train_path = os.path.join(test_dir, "data", "OULA_tra.csv")
        cls.test_path = os.path.join(test_dir, "data", "OULA_test_data.csv")
        
        # Load data once for all tests
        cls.train = pd.read_csv(cls.train_path)
        cls.test = pd.read_csv(cls.test_path)
        cls.X_train = cls.train.drop(columns=['final_result'])
        cls.y_train = cls.train['final_result']
        cls.X_test = cls.test.drop(columns=['final_result'])
        cls.y_test = cls.test['final_result']

    # ----------------------------
    # Decision Tree Tests
    # ----------------------------
    def test_decision_tree_train_and_predict(self):
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        self.assertEqual(len(y_pred), len(self.y_test))
        self.assertTrue(all([p in self.y_test.unique() for p in y_pred]))

    def test_decision_tree_accuracy(self):
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        self.assertGreater(acc, 0)
        self.assertLessEqual(acc, 1)

    # ----------------------------
    # Logistic Regression Tests
    # ----------------------------
    def test_logistic_regression_train_and_predict(self):
        model = LogisticRegression(max_iter=200)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        self.assertEqual(len(y_pred), len(self.y_test))
        self.assertTrue(all([p in self.y_test.unique() for p in y_pred]))

    def test_logistic_regression_accuracy(self):
        model = LogisticRegression(max_iter=200)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        self.assertGreater(acc, 0)
        self.assertLessEqual(acc, 1)

    # ----------------------------
    # Random Forest Tests
    # ----------------------------
    def test_random_forest_train_and_predict(self):
        model = RandomForestClassifier(n_estimators=100)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        self.assertEqual(len(y_pred), len(self.y_test))
        self.assertTrue(all([p in self.y_test.unique() for p in y_pred]))

    def test_random_forest_accuracy(self):
        model = RandomForestClassifier(n_estimators=100)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        self.assertGreater(acc, 0)
        self.assertLessEqual(acc, 1)


if __name__ == "__main__":
    unittest.main()