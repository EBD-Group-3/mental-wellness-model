"""
Secure credential management using environment variables.
This avoids storing sensitive files in repositories.
"""

import os
import json
from google.oauth2 import service_account
from google.cloud import storage


def load_gcs_credentials_from_env():
    """Load GCS credentials from environment variables."""
    try:
        # Check if credentials are provided as environment variables
        creds_json = os.environ.get('GOOGLE_CREDENTIALS_JSON')
        if creds_json:
            # Parse JSON from environment variable
            creds_dict = json.loads(creds_json)
            credentials = service_account.Credentials.from_service_account_info(
                creds_dict,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            return credentials
        
        # Check for individual credential fields
        project_id = os.environ.get('GOOGLE_PROJECT_ID')
        private_key = os.environ.get('GOOGLE_PRIVATE_KEY')
        client_email = os.environ.get('GOOGLE_CLIENT_EMAIL')
        
        if project_id and private_key and client_email:
            creds_dict = {
                "type": "service_account",
                "project_id": project_id,
                "private_key": private_key.replace('\\n', '\n'),
                "client_email": client_email,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"
            }
            credentials = service_account.Credentials.from_service_account_info(
                creds_dict,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            return credentials
        
        return None
        
    except Exception as e:
        print(f"Error loading credentials from environment: {e}")
        return None


def create_gcs_client():
    """Create GCS client with secure credential loading."""
    # Try environment variables first
    credentials = load_gcs_credentials_from_env()
    
    if credentials:
        return storage.Client(credentials=credentials)
    
    # Try default credentials (for deployed environments)
    try:
        return storage.Client()
    except Exception as e:
        print(f"Could not create GCS client: {e}")
        return None


# Example usage
if __name__ == "__main__":
    client = create_gcs_client()
    if client:
        print("✅ GCS client created successfully")
        # Test bucket access
        try:
            bucket = client.bucket("mental_wellness_data_lake")
            if bucket.exists():
                print("✅ Bucket access confirmed")
            else:
                print("❌ Bucket not accessible")
        except Exception as e:
            print(f"❌ Bucket access failed: {e}")
    else:
        print("❌ Could not create GCS client")