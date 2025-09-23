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


class ModelManager:
    """
    Manages model storage, versioning, and organization.
    """
    
    def __init__(self, base_path: str = "/app/models"):
        """
        Initialize the model manager.
        
        Args:
            base_path: Base directory for model storage
        """
        self.base_path = Path(base_path)
        self.production_path = self.base_path / "production"
        self.experiments_path = self.base_path / "experiments" 
        self.backups_path = self.base_path / "backups"
        self.metadata_file = self.base_path / "model_registry.json"
        
        # Create directories
        self._create_directories()
        
        # Load existing metadata
        self.metadata = self._load_metadata()
    
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
        
        return stats