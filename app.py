"""
FastAPI application for Mental Wellness Prediction API.
Provides REST endpoints for training models and making predictions.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import logging
import asyncio
from contextlib import asynccontextmanager

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional
import pandas as pd
import joblib

from mental_wellness_model import MentalWellnessPredictor, DataProcessor, FeatureEngineer
from mental_wellness_model.config import load_config
from model_manager import ModelManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model management
predictor = None
data_processor = None
feature_engineer = None
model_manager = None
model_metadata = {}

class ModelState:
    """Singleton class to manage model state across the API."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.predictor = None
            cls._instance.data_processor = None
            cls._instance.feature_engineer = None
            cls._instance.model_manager = None
            cls._instance.model_metadata = {}
            cls._instance.is_initialized = False
        return cls._instance
    
    def initialize_components(self):
        """Initialize all required components."""
        if not self.is_initialized:
            self.data_processor = DataProcessor()
            self.feature_engineer = FeatureEngineer()
            self.model_manager = ModelManager()
            self.is_initialized = True
            logger.info("Model components initialized successfully")
    
    def get_or_create_predictor(self, model_type: str = 'random_forest'):
        """Get existing predictor or create new one."""
        if self.predictor is None:
            self.predictor = MentalWellnessPredictor(model_type=model_type)
        return self.predictor
    
    def load_model_if_available(self):
        """Try to load any available trained model."""
        if self.predictor is None or not getattr(self.predictor, 'is_trained', False):
            # First, try to load the standard basic_trained_model from GCS
            if self.model_manager and self.model_manager.is_gcs_available():
                try:
                    logger.info("Checking GCS for basic_trained_model...")
                    gcs_models = self.model_manager.list_gcs_models()
                    
                    # Look specifically for basic_trained_model first
                    if "basic_trained_model" in gcs_models:
                        logger.info("Found basic_trained_model in GCS, downloading...")
                        if self.model_manager.download_model_from_gcs("basic_trained_model"):
                            # Try to load the downloaded model
                            local_path = self.model_manager.production_path / "basic_trained_model.joblib"
                            if local_path.exists():
                                self.predictor = MentalWellnessPredictor()
                                self.predictor.load_model(str(local_path))
                                self.model_metadata = {
                                    "model_path": str(local_path),
                                    "loaded_at": datetime.now().isoformat(),
                                    "model_type": getattr(self.predictor, 'model_type', 'unknown'),
                                    "version": "1.0.0",
                                    "source": "gcs_basic_trained_model",
                                    "gcs_model_name": "basic_trained_model"
                                }
                                logger.info("Successfully loaded basic_trained_model from GCS")
                                return True
                    
                    # If basic_trained_model not found, try latest model as fallback
                    elif gcs_models:
                        latest_model = self.model_manager.gcs_storage.get_latest_model()
                        if latest_model:
                            logger.info(f"basic_trained_model not found in GCS, downloading latest: {latest_model}")
                            if self.model_manager.download_model_from_gcs(latest_model):
                                local_path = self.model_manager.production_path / f"{latest_model}.joblib"
                                if local_path.exists():
                                    self.predictor = MentalWellnessPredictor()
                                    self.predictor.load_model(str(local_path))
                                    self.model_metadata = {
                                        "model_path": str(local_path),
                                        "loaded_at": datetime.now().isoformat(),
                                        "model_type": getattr(self.predictor, 'model_type', 'unknown'),
                                        "version": "1.0.0",
                                        "source": "gcs_latest_model",
                                        "gcs_model_name": latest_model
                                    }
                                    logger.info(f"Successfully loaded latest model from GCS: {latest_model}")
                                    return True
                except Exception as gcs_error:
                    logger.warning(f"Failed to load model from GCS: {gcs_error}")
            
            # Try to load from model manager (local production models)
            if self.model_manager:
                production_model_path = self.model_manager.get_production_model_path()
                if production_model_path and os.path.exists(production_model_path):
                    self.predictor = MentalWellnessPredictor()
                    self.predictor.load_model(production_model_path)
                    model_info = self.model_manager.get_model_info(self.model_manager.metadata["current_production"])
                    self.model_metadata = model_info or {
                        "model_path": production_model_path,
                        "loaded_at": datetime.now().isoformat(),
                        "source": "local_production"
                    }
                    logger.info(f"Loaded production model: {production_model_path}")
                    return True
            
            # Try alternative paths with basic_trained_model as priority fallback
            possible_paths = [
                "/app/models/api_trained_model.joblib",
                "./models/api_trained_model.joblib",
                "/app/models/trained_model.joblib",
                "./models/trained_model.joblib",
                "/app/models/basic_trained_model.joblib",
                "./models/basic_trained_model.joblib"
            ]
            
            for model_path in possible_paths:
                if os.path.exists(model_path):
                    try:
                        self.predictor = MentalWellnessPredictor()
                        self.predictor.load_model(model_path)
                        self.model_metadata = {
                            "model_path": model_path,
                            "loaded_at": datetime.now().isoformat(),
                            "model_type": getattr(self.predictor, 'model_type', 'unknown'),
                            "version": "1.0.0",
                            "source": "basic_trained_model" if "basic_trained_model" in model_path else "auto_loaded"
                        }
                        logger.info(f"Successfully loaded model from {model_path}")
                        return True
                    except Exception as e:
                        logger.warning(f"Failed to load model from {model_path}: {e}")
                        continue
        return False

# Initialize global model state
model_state = ModelState()


# Pydantic models for request/response
class IndividualFeatures(BaseModel):
    """Input features for individual prediction."""
    age: float = Field(..., ge=18, le=100, description="Age of the individual")
    sleep_hours: float = Field(..., ge=0, le=24, description="Hours of sleep per night")
    exercise_minutes: float = Field(..., ge=0, le=1440, description="Exercise minutes per week")
    work_stress_level: float = Field(..., ge=1, le=10, description="Work stress level (1-10)")
    mood_rating: float = Field(..., ge=1, le=10, description="Mood rating (1-10)")
    energy_level: float = Field(..., ge=1, le=10, description="Energy level (1-10)")
    avg_heart_rate: float = Field(..., ge=50, le=200, description="Average heart rate (beats per minute)")
    resting_heart_rate: float = Field(..., ge=40, le=120, description="Resting heart rate (beats per minute)")


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    individuals: List[IndividualFeatures] = Field(..., min_items=1, max_items=1000)
    include_metadata: bool = Field(default=True, description="Include prediction metadata")


class PredictionResult(BaseModel):
    """Result for a single prediction."""
    depression: Dict[str, Union[str, float, bool]] = Field(..., description="Depression prediction results")
    anxiety: Dict[str, Union[str, float, bool]] = Field(..., description="Anxiety prediction results")
    onset_day: Dict[str, Union[str, int, float]] = Field(..., description="Mental breakdown onset day prediction")


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    model_info: Dict[str, str] = Field(..., description="Model information")
    predictions: List[PredictionResult] = Field(..., description="Prediction results")
    summary: Dict[str, Union[int, float]] = Field(..., description="Summary statistics")


class TrainingRequest(BaseModel):
    """Request for model training."""
    model_type: str = Field(default="random_forest", description="Type of model to train")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Test set size")
    use_sample_data: bool = Field(default=True, description="Use synthetic sample data")
    sample_size: int = Field(default=1000, ge=100, le=10000, description="Size of sample data")
    data_file: Optional[str] = Field(default=None, description="Path to custom CSV data file (relative to /app/data/)")
    use_gcs_data: bool = Field(default=False, description="Load data from GCS bucket")
    gcs_data_folder: str = Field(default="RawData", description="GCS folder containing the data file")
    gcs_data_filename: str = Field(default="wellness_sample.parquet", description="GCS data filename (supports .csv, .parquet)")

    @validator('model_type')
    def validate_model_type(cls, v):
        if v not in ['random_forest', 'logistic_regression']:
            raise ValueError('model_type must be either "random_forest" or "logistic_regression"')
        return v
    
    @validator('gcs_data_filename')
    def validate_gcs_data_filename(cls, v):
        if not v.lower().endswith(('.csv', '.parquet')):
            raise ValueError('gcs_data_filename must end with .csv or .parquet')
        return v


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_info: Optional[Dict[str, Union[str, bool, List[str]]]] = Field(None, description="Model metadata")


# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # Startup
    logger.info("*** TESTING - Starting Mental Wellness API v2.0 ***")
    logger.info("LIFESPAN: About to call load_default_model()")
    try:
        await load_default_model()
        logger.info("LIFESPAN: load_default_model() completed successfully")
    except Exception as e:
        logger.error(f"LIFESPAN: Error in load_default_model(): {e}")
        import traceback
        logger.error(f"LIFESPAN: Traceback: {traceback.format_exc()}")
    yield
    # Shutdown
    logger.info("Shutting down Mental Wellness API...")


# Initialize FastAPI app
app = FastAPI(
    title="Mental Wellness Prediction API",
    description="REST API for mental wellness risk prediction using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def load_default_model():
    """Load default model if available."""
    global predictor, data_processor, feature_engineer, model_manager, model_metadata
    
    try:
        logger.info("STARTUP: Initializing model state...")
        
        # Initialize components through model state
        model_state.initialize_components()
        
        # Update global variables for backward compatibility
        data_processor = model_state.data_processor
        feature_engineer = model_state.feature_engineer
        model_manager = model_state.model_manager
        
        # Try to load existing model
        if model_state.load_model_if_available():
            predictor = model_state.predictor
            model_metadata = model_state.model_metadata
            logger.info(f"Model loaded successfully during startup from: {model_metadata.get('model_path', 'unknown')}")
        else:
            logger.info("No pre-trained model found during startup. Models will be loaded on-demand when prediction endpoints are called.")
            
    except Exception as e:
        logger.error(f"Error during model loading: {e}")


def get_predictor():
    """Dependency to get the current predictor instance."""
    global predictor, data_processor, feature_engineer, model_metadata
    
    # Ensure components are initialized
    if not model_state.is_initialized:
        model_state.initialize_components()
        data_processor = model_state.data_processor
        feature_engineer = model_state.feature_engineer
    
    # Try to load model if not available
    if model_state.predictor is None or not getattr(model_state.predictor, 'is_trained', False):
        if not model_state.load_model_if_available():
            # If no model could be loaded, try specifically for basic_trained_model
            basic_model_paths = [
                "./models/basic_trained_model.joblib",
                "/app/models/basic_trained_model.joblib",
                "models/basic_trained_model.joblib"
            ]
            
            for model_path in basic_model_paths:
                if os.path.exists(model_path):
                    try:
                        logger.info(f"Attempting to load basic trained model from {model_path}")
                        model_state.predictor = MentalWellnessPredictor()
                        model_state.predictor.load_model(model_path)
                        model_state.model_metadata = {
                            "model_path": model_path,
                            "loaded_at": datetime.now().isoformat(),
                            "model_type": getattr(model_state.predictor, 'model_type', 'unknown'),
                            "version": "1.0.0",
                            "source": "basic_trained_model_fallback"
                        }
                        logger.info(f"Successfully loaded basic trained model from {model_path}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load basic trained model from {model_path}: {e}")
                        continue
        
        predictor = model_state.predictor
        model_metadata = model_state.model_metadata
    
    # Final checks
    if model_state.predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. No trained model found (including basic_trained_model). Please train a model first using /train endpoint."
        )
    
    # Check if predictor has the is_trained attribute and if it's trained
    if hasattr(model_state.predictor, 'is_trained') and not model_state.predictor.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Please train a model first using /train endpoint."
        )
    
    return model_state.predictor


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Mental Wellness Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    # Ensure model state is initialized
    if not model_state.is_initialized:
        model_state.initialize_components()
    
    # Check if model is available and try to load if needed
    if model_state.predictor is None:
        model_state.load_model_if_available()
    
    is_model_loaded = (model_state.predictor is not None and 
                      getattr(model_state.predictor, 'is_trained', False))
    
    # Prepare model info for serialization
    model_info = None
    if model_state.model_metadata:
        model_info = {}
        for key, value in model_state.model_metadata.items():
            if isinstance(value, (str, bool)):
                model_info[key] = value
            elif isinstance(value, list):
                model_info[key] = value  # Keep lists as they are now supported
            else:
                model_info[key] = str(value)  # Convert other types to string
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        model_loaded=is_model_loaded,
        model_info=model_info
    )


@app.post("/train")
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """Train a new mental wellness prediction model."""
    try:
        # Initialize components through model state
        global predictor, data_processor, feature_engineer, model_metadata
        
        # Ensure components are initialized
        model_state.initialize_components()
        
        # Get or create predictor with specified model type
        predictor = model_state.get_or_create_predictor(request.model_type)
        data_processor = model_state.data_processor
        feature_engineer = model_state.feature_engineer
        
        # Load or generate data
        if request.use_sample_data:
            df = data_processor._generate_sample_data(request.sample_size)
            logger.info(f"Generated {len(df)} samples of synthetic data")
        elif request.use_gcs_data:
            # Load data from GCS bucket
            if not model_state.model_manager or not model_state.model_manager.is_gcs_available():
                raise HTTPException(
                    status_code=503,
                    detail="GCS storage not available. Please check your credentials and bucket configuration."
                )
            
            try:
                gcs_storage = model_state.model_manager.gcs_storage
                df = gcs_storage.load_data_from_gcs(
                    data_folder=request.gcs_data_folder,
                    filename=request.gcs_data_filename
                )
                logger.info(f"Loaded {len(df)} samples from GCS: {request.gcs_data_folder}/{request.gcs_data_filename}")
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error loading data from GCS: {str(e)}"
                )
        else:
            # Load custom data from local CSV file
            if not request.data_file:
                raise HTTPException(
                    status_code=400,
                    detail="data_file must be specified when use_sample_data=false and use_gcs_data=false"
                )
            
            # Handle both absolute paths and relative filenames
            if request.data_file.startswith('/app/data/'):
                data_path = request.data_file
            else:
                data_path = f"/app/data/{request.data_file}"
            
            if not os.path.exists(data_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"Data file not found: {data_path}"
                )
            
            try:
                df = data_processor.load_data(data_path=data_path)
                logger.info(f"Loaded {len(df)} samples from custom data file: {request.data_file}")
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error loading data file: {str(e)}"
                )
        
        # Process data
        df = data_processor.clean_data(df)
        if not data_processor.validate_data(df):
            raise HTTPException(status_code=400, detail="Data validation failed")
        
        # Feature engineering
        df = feature_engineer.create_features(df)
        feature_engineer.fit_scaler(df)
        df = feature_engineer.transform_features(df)
        
        # Train model
        results = predictor.train(df, test_size=request.test_size)
        
        # Save model using predictor's built-in save method
        model_path = "/app/models/api_trained_model.joblib"
        predictor.save_model(model_path)
        
        # Update model state
        model_state.predictor = predictor
        model_state.model_metadata = {
            "model_path": model_path,
            "trained_at": datetime.now().isoformat(),
            "model_type": request.model_type,
            "test_size": request.test_size,
            "sample_size": request.sample_size,
            "performance_metrics": results,
            "version": "1.0.0"
        }
        
        # Update global variables for backward compatibility
        model_metadata = model_state.model_metadata
        
        # Upload model to GCS with standard name "basic_trained_model"
        gcs_upload_success = False
        if model_state.model_manager and model_state.model_manager.is_gcs_available():
            try:
                # Use consistent model name for GCS
                model_name = "basic_trained_model"
                
                # Save model through model manager to get proper metadata
                saved_path = model_state.model_manager.save_model(
                    predictor, 
                    model_type="production",
                    model_name=model_name,
                    description=f"API trained {request.model_type} model - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                
                # Upload to GCS with metadata
                gcs_upload_success = model_state.model_manager.upload_model_to_gcs(
                    model_name, 
                    {
                        "training_request": request.dict(),
                        "performance_metrics": results,
                        "api_version": "1.0.0",
                        "trained_at": datetime.now().isoformat(),
                        "model_type": request.model_type
                    }
                )
                
                if gcs_upload_success:
                    logger.info(f"Successfully uploaded model '{model_name}' to GCS")
                    model_state.model_metadata["gcs_uploaded"] = True
                    model_state.model_metadata["gcs_model_name"] = model_name
                    model_metadata = model_state.model_metadata
                    
                    # Also save locally as basic_trained_model for immediate use
                    local_basic_path = "/app/models/basic_trained_model.joblib"
                    try:
                        predictor.save_model(local_basic_path)
                        logger.info(f"Also saved model locally as: {local_basic_path}")
                    except Exception as local_save_error:
                        logger.warning(f"Failed to save local basic model: {local_save_error}")
                else:
                    logger.warning("Failed to upload model to GCS")
            except Exception as gcs_error:
                logger.error(f"Error uploading model to GCS: {gcs_error}")
        else:
            logger.info("GCS not available, model saved locally only")
            # If GCS not available, still save as basic_trained_model locally
            try:
                local_basic_path = "/app/models/basic_trained_model.joblib"
                predictor.save_model(local_basic_path)
                logger.info(f"Saved model locally as: {local_basic_path}")
            except Exception as local_save_error:
                logger.warning(f"Failed to save local basic model: {local_save_error}")
        
        return {
            "message": "Model training completed successfully",
            "results": results,
            "model_metadata": model_metadata
        }
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/predict", response_model=PredictionResult)
async def predict_individual(
    features: IndividualFeatures,
    predictor: MentalWellnessPredictor = Depends(get_predictor)
):
    """Make prediction for an individual."""
    try:
        # Convert to dictionary
        features_dict = features.dict()
        
        # Make prediction
        results = predictor.predict_individual(features_dict)
        
        return PredictionResult(
            depression=results['depression'],
            anxiety=results['anxiety'],
            onset_day=results['onset_day']
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    predictor: MentalWellnessPredictor = Depends(get_predictor)
):
    """Make predictions for multiple individuals."""
    try:
        start_time = datetime.now()
        predictions = []
        high_risk_depression = 0
        high_risk_anxiety = 0
        
        for individual in request.individuals:
            features_dict = individual.dict()
            results = predictor.predict_individual(features_dict)
            
            predictions.append(PredictionResult(
                depression=results['depression'],
                anxiety=results['anxiety'],
                onset_day=results['onset_day']
            ))
            
            # Count high-risk cases
            if results['depression']['risk_level'] in ['High', 'Very High']:
                high_risk_depression += 1
            if results['anxiety']['risk_level'] in ['High', 'Very High']:
                high_risk_anxiety += 1
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = BatchPredictionResponse(
            model_info=model_metadata,
            predictions=predictions,
            summary={
                "total_predictions": len(predictions),
                "high_risk_depression": high_risk_depression,
                "high_risk_anxiety": high_risk_anxiety,
                "processing_time_ms": round(processing_time, 2)
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info")
async def get_model_info():
    """Get information about the currently loaded model."""
    # Ensure model state is initialized
    if not model_state.is_initialized:
        model_state.initialize_components()
    
    # Try to load model if not available
    if model_state.predictor is None:
        model_state.load_model_if_available()
    
    if not model_state.model_metadata:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    # Get additional model manager info if available
    model_manager_info = {}
    if model_state.model_manager:
        try:
            model_manager_info = {
                "available_models": model_state.model_manager.list_models(),
                "production_model": model_state.model_manager.metadata.get("current_production"),
                "model_directory": str(model_state.model_manager.base_path)
            }
        except Exception as e:
            logger.warning(f"Could not get model manager info: {e}")
    
    return {
        "model_metadata": model_state.model_metadata,
        "is_trained": model_state.predictor is not None and getattr(model_state.predictor, 'is_trained', False),
        "feature_columns": model_state.predictor.feature_columns if model_state.predictor else [],
        "model_manager_info": model_manager_info,
        "components_initialized": model_state.is_initialized
    }


@app.delete("/model")
async def unload_model():
    """Unload the current model."""
    global predictor, model_metadata
    
    # Clear model state
    model_state.predictor = None
    model_state.model_metadata = {}
    
    # Clear global variables for backward compatibility
    predictor = None
    model_metadata = {}
    
    return {"message": "Model unloaded successfully"}


@app.get("/models")
async def list_models():
    """List all available models (local and GCS)."""
    if not model_state.model_manager:
        raise HTTPException(status_code=500, detail="Model manager not available")
    
    try:
        local_models = model_state.model_manager.list_models()
        gcs_models = {}
        
        # Get GCS models if available
        if model_state.model_manager.is_gcs_available():
            try:
                gcs_models = model_state.model_manager.list_gcs_models()
            except Exception as gcs_e:
                logger.warning(f"Could not list GCS models: {gcs_e}")
        
        return {
            "local_models": local_models,
            "gcs_models": gcs_models,
            "production_model": model_state.model_manager.metadata.get("current_production"),
            "model_directory": str(model_state.model_manager.base_path),
            "gcs_available": model_state.model_manager.is_gcs_available()
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


@app.post("/models/{model_name}/{version}/promote")
async def promote_model(model_name: str, version: str):
    """Promote a specific model version to production."""
    global predictor, model_metadata
    
    if not model_state.model_manager:
        raise HTTPException(status_code=500, detail="Model manager not available")
    
    try:
        # Promote model
        model_state.model_manager.promote_to_production(model_name, version)
        
        # Load the newly promoted model
        production_model_path = model_state.model_manager.get_production_model_path()
        if production_model_path and os.path.exists(production_model_path):
            model_state.predictor = MentalWellnessPredictor()
            model_state.predictor.load_model(production_model_path)
            model_state.model_metadata = model_state.model_manager.get_model_info(f"{model_name}_{version}")
            
            # Update global variables
            predictor = model_state.predictor
            model_metadata = model_state.model_metadata
            
        return {
            "message": f"Model {model_name} v{version} promoted to production",
            "production_model_path": production_model_path
        }
    except Exception as e:
        logger.error(f"Error promoting model: {e}")
        raise HTTPException(status_code=500, detail=f"Error promoting model: {str(e)}")


@app.post("/models/{model_name}/upload-to-gcs")
async def upload_model_to_gcs(model_name: str):
    """Upload a specific model to Google Cloud Storage."""
    if not model_state.model_manager:
        raise HTTPException(status_code=500, detail="Model manager not available")
    
    if not model_state.model_manager.is_gcs_available():
        raise HTTPException(status_code=503, detail="Google Cloud Storage not available")
    
    try:
        success = model_state.model_manager.upload_model_to_gcs(model_name)
        if success:
            return {
                "message": f"Model {model_name} successfully uploaded to GCS",
                "bucket": "mental_wellness_data_lake",
                "folder": "Model"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to upload model to GCS")
    except Exception as e:
        logger.error(f"Error uploading model to GCS: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading model to GCS: {str(e)}")


@app.post("/models/{model_name}/download-from-gcs")
async def download_model_from_gcs(model_name: str, local_name: str = None):
    """Download a model from Google Cloud Storage."""
    if not model_state.model_manager:
        raise HTTPException(status_code=500, detail="Model manager not available")
    
    if not model_state.model_manager.is_gcs_available():
        raise HTTPException(status_code=503, detail="Google Cloud Storage not available")
    
    try:
        local_name = local_name or model_name
        success = model_state.model_manager.download_model_from_gcs(model_name, local_name)
        if success:
            return {
                "message": f"Model {model_name} successfully downloaded from GCS as {local_name}",
                "local_name": local_name,
                "bucket": "mental_wellness_data_lake",
                "folder": "Model"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to download model from GCS")
    except Exception as e:
        logger.error(f"Error downloading model from GCS: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading model from GCS: {str(e)}")


@app.post("/models/sync-production-to-gcs")
async def sync_production_to_gcs():
    """Sync the current production model to Google Cloud Storage."""
    if not model_state.model_manager:
        raise HTTPException(status_code=500, detail="Model manager not available")
    
    if not model_state.model_manager.is_gcs_available():
        raise HTTPException(status_code=503, detail="Google Cloud Storage not available")
    
    try:
        success = model_state.model_manager.sync_production_model_to_gcs()
        if success:
            production_model = model_state.model_manager.metadata.get("current_production")
            return {
                "message": f"Production model {production_model} successfully synced to GCS",
                "model_name": production_model,
                "bucket": "mental_wellness_data_lake",
                "folder": "Model"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to sync production model to GCS")
    except Exception as e:
        logger.error(f"Error syncing production model to GCS: {e}")
        raise HTTPException(status_code=500, detail=f"Error syncing production model to GCS: {str(e)}")


@app.get("/models/gcs")
async def list_gcs_models():
    """List all models available in Google Cloud Storage."""
    if not model_state.model_manager:
        raise HTTPException(status_code=500, detail="Model manager not available")
    
    if not model_state.model_manager.is_gcs_available():
        raise HTTPException(status_code=503, detail="Google Cloud Storage not available")
    
    try:
        gcs_models = model_state.model_manager.list_gcs_models()
        return {
            "gcs_models": gcs_models,
            "bucket": "mental_wellness_data_lake",
            "folder": "Model",
            "total_models": len(gcs_models)
        }
    except Exception as e:
        logger.error(f"Error listing GCS models: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing GCS models: {str(e)}")


@app.get("/gcs/data")
async def list_gcs_data_files():
    """List available data files in GCS RawData folder."""
    if not model_state.model_manager or not model_state.model_manager.is_gcs_available():
        raise HTTPException(
            status_code=503,
            detail="GCS storage not available. Please check your credentials and bucket configuration."
        )
    
    try:
        gcs_storage = model_state.model_manager.gcs_storage
        client = gcs_storage.client
        bucket = gcs_storage.bucket
        
        # List files in RawData folder
        data_files = []
        blobs = client.list_blobs(bucket, prefix="RawData/")
        
        for blob in blobs:
            if not blob.name.endswith('/'):  # Skip folder markers
                file_info = {
                    "name": blob.name.split('/')[-1],
                    "full_path": blob.name,
                    "size_bytes": blob.size,
                    "updated": blob.updated.isoformat() if blob.updated else None,
                    "content_type": blob.content_type
                }
                data_files.append(file_info)
        
        return {
            "gcs_data_files": data_files,
            "bucket": "mental_wellness_data_lake",
            "folder": "RawData",
            "total_files": len(data_files)
        }
    except Exception as e:
        logger.error(f"Error listing GCS data files: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing GCS data files: {str(e)}")


@app.get("/gcs/data/preview")
async def preview_gcs_data(
    filename: str = "wellness_sample.parquet",
    folder: str = "RawData",
    rows: int = 5
):
    """Preview data from GCS file."""
    if not model_state.model_manager or not model_state.model_manager.is_gcs_available():
        raise HTTPException(
            status_code=503,
            detail="GCS storage not available. Please check your credentials and bucket configuration."
        )
    
    try:
        gcs_storage = model_state.model_manager.gcs_storage
        df = gcs_storage.load_data_from_gcs(data_folder=folder, filename=filename)
        
        # Get basic info about the dataset
        preview_data = {
            "filename": filename,
            "folder": folder,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "preview_rows": df.head(rows).to_dict('records'),
            "sample_statistics": df.describe().to_dict() if len(df) > 0 else {}
        }
        
        return preview_data
        
    except Exception as e:
        logger.error(f"Error previewing GCS data: {e}")
        raise HTTPException(status_code=400, detail=f"Error previewing GCS data: {str(e)}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except ImportError:
        print("uvicorn not installed. Install with: pip install uvicorn")
        print("Or run with: gunicorn app:app -c gunicorn.conf.py")