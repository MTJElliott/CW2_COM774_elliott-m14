import unittest
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

class TestAzureConnection(unittest.TestCase):

    def setUp(self):
        # These environment variables need to be set for the test
        import os
        self.subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
        self.resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
        self.workspace = os.environ.get("AZURE_ML_WORKSPACE")

    def test_credentials_exist(self):
        self.assertIsNotNone(self.subscription_id, "AZURE_SUBSCRIPTION_ID is not set")
        self.assertIsNotNone(self.resource_group, "AZURE_RESOURCE_GROUP is not set")
        self.assertIsNotNone(self.workspace, "AZURE_ML_WORKSPACE is not set")

    def test_ml_client_connection(self):
        credential = DefaultAzureCredential()
        client = MLClient(credential, self.subscription_id, self.resource_group, self.workspace)
        # basic property check
        self.assertEqual(client.workspace_name, self.workspace)

if __name__ == "__main__":
    unittest.main()