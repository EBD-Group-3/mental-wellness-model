"""
Google Cloud Storage integration for model persistence.
Handles uploading and downloading models to/from GCS bucket.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from google.cloud import storage
from google.auth import default
from google.oauth2 import service_account
import joblib
import tempfile
import pandas as pd
from secure_credential_manager import create_secure_gcs_client


logger = logging.getLogger(__name__)


class GCSModelStorage:
    """
    Google Cloud Storage integration for model persistence.
    """
    
    def __init__(self, 
                 bucket_name: str = "mental_wellness_data_lake",
                 model_folder: str = "Model",
                 credentials_path: Optional[str] = None):
        """
        Initialize GCS model storage.
        
        Args:
            bucket_name: Name of the GCS bucket
            model_folder: Folder name in the bucket for models
            credentials_path: Path to service account credentials JSON file (deprecated, use environment variables)
        """
        self.bucket_name = bucket_name
        self.model_folder = model_folder
        # Credentials path is deprecated - now using environment variables via secure_credential_manager
        self.credentials_path = credentials_path
        
        # Initialize GCS client
        self.client = None
        self.bucket = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Google Cloud Storage client using secure credential manager."""
        try:
            # Use secure credential manager
            self.client = create_secure_gcs_client()
            
            if self.client:
                # Get bucket reference
                self.bucket = self.client.bucket(self.bucket_name)
                logger.info(f"✅ Successfully initialized secure GCS client for bucket: {self.bucket_name}")
            else:
                logger.error("❌ Failed to create secure GCS client")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize GCS client: {e}")
            self.client = None
            self.bucket = None
    
    def is_available(self) -> bool:
        """Check if GCS is available and properly configured."""
        return self.client is not None and self.bucket is not None
    
    def upload_model(self, local_model_path: str, model_name: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Upload a model file to GCS.
        
        Args:
            local_model_path: Local path to the model file
            model_name: Name for the model in GCS (without extension)
            metadata: Additional metadata to store with the model
            
        Returns:
            True if upload successful, False otherwise
        """
        if not self.is_available():
            logger.error("GCS not available for model upload")
            return False
        
        if not os.path.exists(local_model_path):
            logger.error(f"Local model file not found: {local_model_path}")
            return False
        
        try:
            # Upload model file
            model_blob_name = f"{self.model_folder}/{model_name}.joblib"
            model_blob = self.bucket.blob(model_blob_name)
            
            # Add metadata
            if metadata:
                model_blob.metadata = {
                    **metadata,
                    'uploaded_at': datetime.now().isoformat(),
                    'upload_source': 'mental_wellness_api'
                }
            
            # Upload the file
            logger.info(f"Uploading model to GCS: {model_blob_name}")
            model_blob.upload_from_filename(local_model_path)
            
            # Also upload metadata as separate JSON file
            if metadata:
                metadata_blob_name = f"{self.model_folder}/{model_name}_metadata.json"
                metadata_blob = self.bucket.blob(metadata_blob_name)
                metadata_json = json.dumps({
                    **metadata,
                    'uploaded_at': datetime.now().isoformat(),
                    'model_file': model_blob_name
                }, indent=2)
                metadata_blob.upload_from_string(metadata_json, content_type='application/json')
                logger.info(f"Uploaded model metadata to GCS: {metadata_blob_name}")
            
            logger.info(f"Successfully uploaded model {model_name} to GCS")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload model to GCS: {e}")
            return False
    
    def download_model(self, model_name: str, local_path: str) -> Optional[Dict[str, Any]]:
        """
        Download a model file from GCS.
        
        Args:
            model_name: Name of the model in GCS (without extension)
            local_path: Local path where to save the model
            
        Returns:
            Model metadata if successful, None otherwise
        """
        if not self.is_available():
            logger.error("GCS not available for model download")
            return None
        
        try:
            # Download model file
            model_blob_name = f"{self.model_folder}/{model_name}.joblib"
            model_blob = self.bucket.blob(model_blob_name)
            
            if not model_blob.exists():
                logger.warning(f"Model not found in GCS: {model_blob_name}")
                return None
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download the model file
            logger.info(f"Downloading model from GCS: {model_blob_name}")
            model_blob.download_to_filename(local_path)
            
            # Try to download metadata
            metadata = {}
            try:
                metadata_blob_name = f"{self.model_folder}/{model_name}_metadata.json"
                metadata_blob = self.bucket.blob(metadata_blob_name)
                if metadata_blob.exists():
                    metadata_json = metadata_blob.download_as_text()
                    metadata = json.loads(metadata_json)
                    logger.info(f"Downloaded model metadata from GCS: {metadata_blob_name}")
            except Exception as meta_e:
                logger.warning(f"Could not download metadata: {meta_e}")
            
            # Add download info to metadata
            metadata.update({
                'downloaded_at': datetime.now().isoformat(),
                'local_path': local_path,
                'gcs_path': model_blob_name
            })
            
            logger.info(f"Successfully downloaded model {model_name} from GCS")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to download model from GCS: {e}")
            return None
    
    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all models available in GCS.
        
        Returns:
            Dictionary mapping model names to their metadata
        """
        if not self.is_available():
            logger.error("GCS not available for listing models")
            return {}
        
        try:
            models = {}
            blobs = self.client.list_blobs(self.bucket, prefix=f"{self.model_folder}/")
            
            for blob in blobs:
                if blob.name.endswith('.joblib'):
                    # Extract model name
                    model_name = blob.name.replace(f"{self.model_folder}/", "").replace(".joblib", "")
                    
                    # Get basic info
                    models[model_name] = {
                        'name': model_name,
                        'size_bytes': blob.size,
                        'created': blob.time_created.isoformat() if blob.time_created else None,
                        'updated': blob.updated.isoformat() if blob.updated else None,
                        'gcs_path': blob.name,
                        'metadata': blob.metadata or {}
                    }
            
            logger.info(f"Found {len(models)} models in GCS")
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models from GCS: {e}")
            return {}
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model from GCS.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            True if deletion successful, False otherwise
        """
        if not self.is_available():
            logger.error("GCS not available for model deletion")
            return False
        
        try:
            # Delete model file
            model_blob_name = f"{self.model_folder}/{model_name}.joblib"
            model_blob = self.bucket.blob(model_blob_name)
            if model_blob.exists():
                model_blob.delete()
                logger.info(f"Deleted model file: {model_blob_name}")
            
            # Delete metadata file
            metadata_blob_name = f"{self.model_folder}/{model_name}_metadata.json"
            metadata_blob = self.bucket.blob(metadata_blob_name)
            if metadata_blob.exists():
                metadata_blob.delete()
                logger.info(f"Deleted metadata file: {metadata_blob_name}")
            
            logger.info(f"Successfully deleted model {model_name} from GCS")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model from GCS: {e}")
            return False
    
    def get_latest_model(self) -> Optional[str]:
        """
        Get the name of the most recently uploaded model.
        
        Returns:
            Name of the latest model, or None if no models found
        """
        models = self.list_models()
        if not models:
            return None
        
        # Sort by creation time
        latest_model = max(models.items(), key=lambda x: x[1].get('created', ''))
        return latest_model[0] if latest_model else None
    
    def sync_model_to_gcs(self, local_model_path: str, model_name: str, 
                         force_upload: bool = False, metadata: Dict[str, Any] = None) -> bool:
        """
        Sync a local model to GCS, uploading only if needed.
        
        Args:
            local_model_path: Path to local model file
            model_name: Name for the model in GCS
            force_upload: If True, upload even if model exists in GCS
            metadata: Metadata to store with the model
            
        Returns:
            True if sync successful, False otherwise
        """
        if not self.is_available():
            return False
        
        # Check if model exists in GCS
        model_blob_name = f"{self.model_folder}/{model_name}.joblib"
        model_blob = self.bucket.blob(model_blob_name)
        
        if model_blob.exists() and not force_upload:
            logger.info(f"Model {model_name} already exists in GCS, skipping upload")
            return True
        
        return self.upload_model(local_model_path, model_name, metadata)
    
    def download_data_file(self, data_folder: str, filename: str, local_path: str) -> bool:
        """
        Download a data file from GCS.
        
        Args:
            data_folder: Folder in GCS bucket containing the data file
            filename: Name of the data file to download
            local_path: Local path where to save the downloaded file
            
        Returns:
            True if download successful, False otherwise
        """
        if not self.is_available():
            logger.error("GCS not available for data download")
            return False
        
        try:
            # Construct blob path
            blob_path = f"{data_folder}/{filename}"
            blob = self.bucket.blob(blob_path)
            
            if not blob.exists():
                logger.error(f"Data file not found in GCS: {blob_path}")
                return False
            
            # Create local directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download the file
            blob.download_to_filename(local_path)
            logger.info(f"✅ Downloaded data file from GCS: {blob_path} → {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to download data file from GCS: {e}")
            return False
    
    def load_data_from_gcs(self, data_folder: str, filename: str) -> pd.DataFrame:
        """
        Load data directly from GCS into a pandas DataFrame.
        
        Args:
            data_folder: Folder in GCS bucket containing the data file
            filename: Name of the data file (supports .csv, .parquet)
            
        Returns:
            DataFrame with the loaded data
            
        Raises:
            Exception if file cannot be loaded
        """
        if not self.is_available():
            raise Exception("GCS not available for data loading")
        
        try:
            # Construct blob path
            blob_path = f"{data_folder}/{filename}"
            blob = self.bucket.blob(blob_path)
            
            if not blob.exists():
                raise Exception(f"Data file not found in GCS: {blob_path}")
            
            # Download data to memory
            data_bytes = blob.download_as_bytes()
            
            # Load based on file extension
            if filename.lower().endswith('.csv'):
                from io import StringIO
                data_str = data_bytes.decode('utf-8')
                df = pd.read_csv(StringIO(data_str))
            elif filename.lower().endswith('.parquet'):
                from io import BytesIO
                df = pd.read_parquet(BytesIO(data_bytes))
            else:
                raise Exception(f"Unsupported file format: {filename}. Supported formats: .csv, .parquet")
            
            logger.info(f"✅ Loaded {len(df)} rows from GCS: {blob_path}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load data from GCS: {e}")
            raise