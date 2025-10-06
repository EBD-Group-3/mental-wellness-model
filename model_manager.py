"""
Enhanced Model Management for Mental Wellness API
Provides organized model storage and versioning capabilities.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import shutil
import logging
from gcs_model_storage import GCSModelStorage


class ModelManager:
    """
    Manages model storage, versioning, and organization.
    """
    
    def __init__(self, base_path: str = "/app/models", enable_gcs: bool = True):
        """
        Initialize the model manager.
        
        Args:
            base_path: Base directory for model storage
            enable_gcs: Whether to enable Google Cloud Storage integration
        """
        self.logger = logging.getLogger(__name__)
        self.base_path = Path(base_path)
        self.production_path = self.base_path / "production"
        self.experiments_path = self.base_path / "experiments" 
        self.backups_path = self.base_path / "backups"
        self.metadata_file = self.base_path / "model_registry.json"
        
        # Initialize GCS storage
        self.gcs_storage = None
        if enable_gcs:
            try:
                self.gcs_storage = GCSModelStorage()
                if self.gcs_storage.is_available():
                    self.logger.info("GCS model storage initialized successfully")
                else:
                    self.logger.warning("GCS model storage not available")
                    self.gcs_storage = None
            except Exception as e:
                self.logger.error(f"Failed to initialize GCS storage: {e}")
                self.gcs_storage = None
        
        # Create directories
        self._create_directories()
        
        # Load existing metadata
        self.metadata = self._load_metadata()
        
        # Sync with GCS on initialization
        if self.gcs_storage:
            self._sync_with_gcs_on_startup()
    
    def _create_directories(self):
        """Create necessary directories."""
        for path in [self.production_path, self.experiments_path, self.backups_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def _load_metadata(self) -> Dict:
        """Load model metadata from registry."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"models": [], "current_production": None}
    
    def _save_metadata(self):
        """Save model metadata to registry."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_model(self, predictor, model_type: str = "experiment", 
                   model_name: Optional[str] = None, 
                   description: Optional[str] = None) -> str:
        """
        Save a model with organized structure.
        
        Args:
            predictor: The trained predictor instance
            model_type: "production", "experiment", or "backup"
            model_name: Optional custom name for the model
            description: Optional description
            
        Returns:
            Path where model was saved
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if model_name is None:
            model_name = f"model_{timestamp}"
        
        # Determine save path based on type
        if model_type == "production":
            model_path = self.production_path / f"{model_name}.joblib"
        elif model_type == "backup":
            model_path = self.backups_path / f"{model_name}.joblib"
        else:  # experiment
            model_path = self.experiments_path / f"{model_name}.joblib"
        
        # Save the model
        predictor.save_model(str(model_path))
        
        # Update metadata
        model_info = {
            "name": model_name,
            "path": str(model_path),
            "type": model_type,
            "created_at": datetime.now().isoformat(),
            "model_type": getattr(predictor, 'model_type', 'unknown'),
            "description": description or f"Model saved as {model_type}",
            "is_trained": getattr(predictor, 'is_trained', False),
            "feature_columns": getattr(predictor, 'feature_columns', [])
        }
        
        self.metadata["models"].append(model_info)
        
        # If this is a production model, update current production
        if model_type == "production":
            self.metadata["current_production"] = model_name
        
        self._save_metadata()
        
        return str(model_path)
    
    def promote_to_production(self, experiment_name: str, 
                            production_name: Optional[str] = None) -> str:
        """
        Promote an experimental model to production.
        
        Args:
            experiment_name: Name of experiment model to promote
            production_name: Optional name for production model
            
        Returns:
            Path of promoted production model
        """
        # Find experiment model
        experiment_model = None
        for model in self.metadata["models"]:
            if model["name"] == experiment_name and model["type"] == "experiment":
                experiment_model = model
                break
        
        if not experiment_model:
            raise ValueError(f"Experiment model '{experiment_name}' not found")
        
        # Backup current production model if exists
        if self.metadata["current_production"]:
            self._backup_current_production()
        
        # Copy experiment to production
        if production_name is None:
            production_name = f"production_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        experiment_path = Path(experiment_model["path"])
        production_path = self.production_path / f"{production_name}.joblib"
        
        shutil.copy2(experiment_path, production_path)
        
        # Update metadata
        production_info = experiment_model.copy()
        production_info.update({
            "name": production_name,
            "path": str(production_path),
            "type": "production",
            "promoted_at": datetime.now().isoformat(),
            "promoted_from": experiment_name
        })
        
        self.metadata["models"].append(production_info)
        self.metadata["current_production"] = production_name
        self._save_metadata()
        
        return str(production_path)
    
    def _backup_current_production(self):
        """Backup current production model."""
        if not self.metadata["current_production"]:
            return
        
        # Find current production model
        for model in self.metadata["models"]:
            if (model["name"] == self.metadata["current_production"] and 
                model["type"] == "production"):
                
                production_path = Path(model["path"])
                if production_path.exists():
                    backup_name = f"backup_{model['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    backup_path = self.backups_path / f"{backup_name}.joblib"
                    shutil.copy2(production_path, backup_path)
                    
                    # Add backup to metadata
                    backup_info = model.copy()
                    backup_info.update({
                        "name": backup_name,
                        "path": str(backup_path),
                        "type": "backup",
                        "backed_up_at": datetime.now().isoformat()
                    })
                    self.metadata["models"].append(backup_info)
                break
    
    def list_models(self, model_type: Optional[str] = None) -> List[Dict]:
        """
        List available models.
        
        Args:
            model_type: Optional filter by type ("production", "experiment", "backup")
            
        Returns:
            List of model information
        """
        if model_type:
            return [m for m in self.metadata["models"] if m["type"] == model_type]
        return self.metadata["models"]
    
    def get_production_model_path(self) -> Optional[str]:
        """Get path to current production model."""
        if not self.metadata["current_production"]:
            return None
        
        for model in self.metadata["models"]:
            if (model["name"] == self.metadata["current_production"] and 
                model["type"] == "production"):
                return model["path"]
        return None
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model and its metadata.
        
        Args:
            model_name: Name of model to delete
            
        Returns:
            True if deleted successfully
        """
        model_to_delete = None
        model_index = -1
        
        for i, model in enumerate(self.metadata["models"]):
            if model["name"] == model_name:
                model_to_delete = model
                model_index = i
                break
        
        if not model_to_delete:
            return False
        
        # Don't delete current production model
        if (model_to_delete["type"] == "production" and 
            self.metadata["current_production"] == model_name):
            raise ValueError("Cannot delete current production model")
        
        # Delete file
        model_path = Path(model_to_delete["path"])
        if model_path.exists():
            model_path.unlink()
        
        # Remove from metadata
        self.metadata["models"].pop(model_index)
        self._save_metadata()
        
        return True
    
    def cleanup_old_models(self, keep_last: int = 5, model_type: str = "experiment"):
        """
        Clean up old models, keeping only the most recent ones.
        
        Args:
            keep_last: Number of recent models to keep
            model_type: Type of models to clean up
        """
        models = [m for m in self.metadata["models"] if m["type"] == model_type]
        models.sort(key=lambda x: x["created_at"], reverse=True)
        
        models_to_delete = models[keep_last:]
        
        for model in models_to_delete:
            try:
                self.delete_model(model["name"])
            except Exception as e:
                print(f"Error deleting model {model['name']}: {e}")
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get detailed information about a specific model."""
        for model in self.metadata["models"]:
            if model["name"] == model_name:
                return model
        return None
    
    def get_storage_stats(self) -> Dict:
        """Get storage statistics for models."""
        stats = {
            "total_models": len(self.metadata["models"]),
            "production_models": len([m for m in self.metadata["models"] if m["type"] == "production"]),
            "experiment_models": len([m for m in self.metadata["models"] if m["type"] == "experiment"]),
            "backup_models": len([m for m in self.metadata["models"] if m["type"] == "backup"]),
            "current_production": self.metadata["current_production"],
            "storage_paths": {
                "production": str(self.production_path),
                "experiments": str(self.experiments_path),
                "backups": str(self.backups_path)
            }
        }
        
        # Add GCS stats if available
        if self.gcs_storage and self.gcs_storage.is_available():
            try:
                gcs_models = self.gcs_storage.list_models()
                stats["gcs_models"] = len(gcs_models)
                stats["gcs_available"] = True
            except Exception as e:
                self.logger.warning(f"Could not get GCS stats: {e}")
                stats["gcs_models"] = 0
                stats["gcs_available"] = False
        else:
            stats["gcs_models"] = 0
            stats["gcs_available"] = False
        
        return stats
    
    def _sync_with_gcs_on_startup(self):
        """Sync with GCS on startup - download latest models if local storage is empty."""
        try:
            if not self.gcs_storage or not self.gcs_storage.is_available():
                return
            
            # Check if we have any local models
            local_models = self.list_models()
            if local_models:
                self.logger.info("Local models found, skipping GCS sync on startup")
                return
            
            # Get models from GCS
            gcs_models = self.gcs_storage.list_models()
            if not gcs_models:
                self.logger.info("No models found in GCS")
                return
            
            # Download the latest model from GCS
            latest_model = self.gcs_storage.get_latest_model()
            if latest_model:
                local_path = self.production_path / f"{latest_model}.joblib"
                metadata = self.gcs_storage.download_model(latest_model, str(local_path))
                
                if metadata and os.path.exists(local_path):
                    # Add to local metadata
                    model_info = {
                        "name": latest_model,
                        "path": str(local_path),
                        "type": "production",
                        "created_at": metadata.get('uploaded_at', datetime.now().isoformat()),
                        "description": f"Downloaded from GCS: {latest_model}",
                        "gcs_synced": True,
                        "gcs_metadata": metadata
                    }
                    
                    self.metadata["models"].append(model_info)
                    self.metadata["current_production"] = latest_model
                    self._save_metadata()
                    
                    self.logger.info(f"Downloaded and set production model from GCS: {latest_model}")
        
        except Exception as e:
            self.logger.error(f"Failed to sync with GCS on startup: {e}")
    
    def upload_model_to_gcs(self, model_name: str, metadata: Dict = None) -> bool:
        """
        Upload a specific model to GCS.
        
        Args:
            model_name: Name of the model to upload
            metadata: Additional metadata to store
            
        Returns:
            True if successful, False otherwise
        """
        if not self.gcs_storage or not self.gcs_storage.is_available():
            self.logger.warning("GCS storage not available")
            return False
        
        # Find model info
        model_info = self.get_model_info(model_name)
        if not model_info:
            self.logger.error(f"Model {model_name} not found in local registry")
            return False
        
        model_path = model_info["path"]
        if not os.path.exists(model_path):
            self.logger.error(f"Model file not found: {model_path}")
            return False
        
        # Prepare metadata
        upload_metadata = {
            "model_name": model_name,
            "model_type": model_info.get("type", "unknown"),
            "created_at": model_info.get("created_at"),
            "description": model_info.get("description", ""),
            "local_path": model_path,
            "version": model_info.get("version", "1.0.0")
        }
        
        if metadata:
            upload_metadata.update(metadata)
        
        # Upload to GCS
        success = self.gcs_storage.upload_model(model_path, model_name, upload_metadata)
        
        if success:
            # Update local metadata to mark as GCS synced
            model_info["gcs_synced"] = True
            model_info["gcs_uploaded_at"] = datetime.now().isoformat()
            self._save_metadata()
            self.logger.info(f"Successfully uploaded model {model_name} to GCS")
        
        return success
    
    def download_model_from_gcs(self, model_name: str, local_name: str = None) -> bool:
        """
        Download a model from GCS.
        
        Args:
            model_name: Name of the model in GCS
            local_name: Local name for the model (defaults to GCS name)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.gcs_storage or not self.gcs_storage.is_available():
            self.logger.warning("GCS storage not available")
            return False
        
        local_name = local_name or model_name
        local_path = self.production_path / f"{local_name}.joblib"
        
        # Download from GCS
        metadata = self.gcs_storage.download_model(model_name, str(local_path))
        
        if metadata and os.path.exists(local_path):
            # Add to local metadata
            model_info = {
                "name": local_name,
                "path": str(local_path),
                "type": "production",
                "created_at": metadata.get('uploaded_at', datetime.now().isoformat()),
                "description": f"Downloaded from GCS: {model_name}",
                "gcs_synced": True,
                "gcs_metadata": metadata,
                "downloaded_at": datetime.now().isoformat()
            }
            
            # Remove existing model with same name
            self.metadata["models"] = [m for m in self.metadata["models"] if m["name"] != local_name]
            self.metadata["models"].append(model_info)
            self._save_metadata()
            
            self.logger.info(f"Successfully downloaded model {model_name} from GCS as {local_name}")
            return True
        
        return False
    
    def sync_production_model_to_gcs(self) -> bool:
        """
        Sync the current production model to GCS.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.metadata["current_production"]:
            self.logger.warning("No production model to sync")
            return False
        
        return self.upload_model_to_gcs(
            self.metadata["current_production"],
            {"sync_type": "production_sync", "auto_sync": True}
        )
    
    def list_gcs_models(self) -> Dict[str, Dict]:
        """
        List all models available in GCS.
        
        Returns:
            Dictionary of models in GCS
        """
        if not self.gcs_storage or not self.gcs_storage.is_available():
            return {}
        
        return self.gcs_storage.list_models()
    
    def is_gcs_available(self) -> bool:
        """Check if GCS storage is available."""
        return self.gcs_storage is not None and self.gcs_storage.is_available()