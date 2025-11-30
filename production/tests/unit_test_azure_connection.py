import unittest
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient

class TestAzureConnection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.subscription_id = "21a87bf5-6e2d-4b25-9722-6537add69371"
        cls.resource_group = "elliott-m14-CW2"  
        cls.workspace = "CW2-COM774_elliott"
        cls.tenant_id = "6f0b9487-4fa8-42a8-aeb4-bf2e2c22d4e8"
        cls.client_id = "e7d29922-c732-4c6f-bdab-ac22f06ef5c8"
        cls.client_secret = "wZy8Q~vhpO2h_n9.wUCj6H7QLfE2MJVRNxGQdchB"

    def test_ml_client_connection(self):
        # Use ClientSecretCredential for explicit credentials
        credential = ClientSecretCredential(
            tenant_id=self.tenant_id,
            client_id=self.client_id,
            client_secret=self.client_secret
        )

        client = MLClient(credential, self.subscription_id, self.resource_group, self.workspace)
        
        # Basic property check
        self.assertEqual(client.workspace_name, self.workspace)

if __name__ == "__main__":
    unittest.main()