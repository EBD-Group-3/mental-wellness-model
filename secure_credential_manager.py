"""
Secure credential loading for production deployment.
Supports multiple credential sources with fallbacks.
"""

import os
import json
import logging
from pathlib import Path
from google.oauth2 import service_account
from google.cloud import storage

logger = logging.getLogger(__name__)


class SecureGCSCredentials:
    """Secure GCS credential management."""
    
    @staticmethod
    def load_from_env_json():
        """Load credentials from GOOGLE_CREDENTIALS_JSON environment variable."""
        try:
            creds_json = os.environ.get('GOOGLE_CREDENTIALS_JSON')
            if creds_json:
                creds_dict = json.loads(creds_json)
                return service_account.Credentials.from_service_account_info(
                    creds_dict,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
        except Exception as e:
            logger.warning(f"Failed to load credentials from env JSON: {e}")
        return None
    
    @staticmethod
    def load_from_env_fields():
        """Load credentials from individual environment variables."""
        try:
            required_fields = {
                'project_id': os.environ.get('GCS_PROJECT_ID'),
                'private_key': os.environ.get('GCS_PRIVATE_KEY', '').replace('\\n', '\n'),
                'client_email': os.environ.get('GCS_CLIENT_EMAIL'),
                'private_key_id': os.environ.get('GCS_PRIVATE_KEY_ID'),
            }
            
            if all(required_fields.values()):
                creds_dict = {
                    "type": "service_account",
                    **required_fields,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs"
                }
                
                return service_account.Credentials.from_service_account_info(
                    creds_dict,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
        except Exception as e:
            logger.warning(f"Failed to load credentials from env fields: {e}")
        return None
    
    @staticmethod
    def load_from_file():
        """Load credentials from local file (development only)."""
        try:
            # Only try this in development
            if os.environ.get('ENVIRONMENT') == 'development':
                creds_path = Path("credentials") / "mentalwellness-473814-key.json"
                if creds_path.exists():
                    return service_account.Credentials.from_service_account_file(
                        str(creds_path),
                        scopes=['https://www.googleapis.com/auth/cloud-platform']
                    )
        except Exception as e:
            logger.warning(f"Failed to load credentials from file: {e}")
        return None
    
    @staticmethod
    def load_default():
        """Load default credentials (for GCP environments)."""
        try:
            # This works in Google Cloud environments (Cloud Run, GKE, etc.)
            from google.auth import default
            credentials, project = default(
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            return credentials
        except Exception as e:
            logger.warning(f"Failed to load default credentials: {e}")
        return None
    
    @classmethod
    def get_credentials(cls):
        """Get credentials using the first available method."""
        # For production/cloud environments, prioritize environment variables
        if os.environ.get('ENVIRONMENT') == 'production' or os.environ.get('RENDER'):
            methods = [
                ("Environment JSON", cls.load_from_env_json),
                ("Environment Fields", cls.load_from_env_fields),
                ("Default", cls.load_default),
                ("Local File", cls.load_from_file)
            ]
        else:
            # For development, try file first
            methods = [
                ("Environment JSON", cls.load_from_env_json),
                ("Environment Fields", cls.load_from_env_fields),
                ("Local File", cls.load_from_file),
                ("Default", cls.load_default)
            ]
        
        for method_name, method in methods:
            try:
                credentials = method()
                if credentials:
                    logger.info(f"✅ Loaded credentials using: {method_name}")
                    return credentials
            except Exception as e:
                logger.warning(f"❌ {method_name} failed: {e}")
        
        logger.error("❌ No valid credentials found")
        return None


def create_secure_gcs_client():
    """Create a GCS client with secure credential loading."""
    credentials = SecureGCSCredentials.get_credentials()
    
    if credentials:
        try:
            client = storage.Client(credentials=credentials)
            # Test the client
            list(client.list_buckets(max_results=1))
            return client
        except Exception as e:
            logger.error(f"Failed to create working GCS client: {e}")
    
    return None


# Example environment variable setup for production
def print_env_setup_instructions():
    """Print instructions for setting up environment variables."""
    print("""
🔧 Environment Variable Setup Instructions:

For Render Deployment (Recommended):
===================================

In your Render service dashboard, set these environment variables:

Option 1: JSON in single environment variable (Preferred)
GOOGLE_CREDENTIALS_JSON={"type": "service_account", "project_id": "mentalwellness-473814", ...}

Option 2: Individual fields
GCS_PROJECT_ID=mentalwellness-473814
GCS_PRIVATE_KEY=-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n
GCS_CLIENT_EMAIL=your-service-account@mentalwellness-473814.iam.gserviceaccount.com
GCS_PRIVATE_KEY_ID=your-private-key-id

Additional variables:
ENVIRONMENT=production

For Local Development:
=====================
export GOOGLE_CREDENTIALS_JSON='{"type": "service_account", "project_id": "mentalwellness-473814", ...}'

For Docker:
===========
docker run -e GOOGLE_CREDENTIALS_JSON='{"type":"service_account",...}' your-app

For Kubernetes:
===============
kubectl create secret generic gcs-credentials --from-literal=GOOGLE_CREDENTIALS_JSON='{"type":"service_account",...}'

For Cloud Run/GKE:
==================
Uses default service account - no manual setup needed

📖 See RENDER_DEPLOYMENT_GUIDE.md for complete deployment instructions.
    """)


if __name__ == "__main__":
    print("🔐 Testing Secure GCS Authentication")
    
    client = create_secure_gcs_client()
    
    if client:
        print("✅ Secure GCS client created successfully")
        
        # Test bucket access
        try:
            bucket = client.bucket("mental_wellness_data_lake")
            if bucket.exists():
                print("✅ Bucket access confirmed")
            else:
                print("❌ Bucket not found or not accessible")
        except Exception as e:
            print(f"❌ Bucket access test failed: {e}")
    else:
        print("❌ Could not create secure GCS client")
        print("\n" + "="*50)
        print_env_setup_instructions()