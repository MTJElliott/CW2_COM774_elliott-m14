import unittest
import pandas as pd
import os

class TestDataInput(unittest.TestCase):
    
    def setUp(self):
        self.train_path = "data/oula_train_pt1.csv"
        self.test_path = "data/oula_test.csv"

    def test_train_file_exists(self):
        self.assertTrue(os.path.exists(self.train_path), "Training data does not exist")

    def test_test_file_exists(self):
        self.assertTrue(os.path.exists(self.test_path), "Testing data does not exist")

    def test_columns_exist(self):
        required_columns = ['final_result', 'gender']  # Add more if needed
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)
        for col in required_columns:
            self.assertIn(col, train.columns)
            self.assertIn(col, test.columns)

    def test_data_not_empty(self):
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)
        self.assertGreater(len(train), 0, "Training data is empty")
        self.assertGreater(len(test), 0, "Testing data is empty")

if __name__ == "__main__":
    unittest.main()