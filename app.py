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
import pandas as pd
import joblib

from mental_wellness_model import MentalWellnessPredictor, DataProcessor, FeatureEngineer
from mental_wellness_model.config import load_config


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
model_metadata = {}


# Pydantic models for request/response
class IndividualFeatures(BaseModel):
    """Input features for individual prediction."""
    age: float = Field(..., ge=18, le=100, description="Age of the individual")
    sleep_hours: float = Field(..., ge=0, le=24, description="Hours of sleep per night")
    exercise_frequency: int = Field(..., ge=0, le=7, description="Exercise sessions per week")
    social_interaction_score: float = Field(..., ge=1, le=10, description="Social interaction rating (1-10)")
    work_stress_level: float = Field(..., ge=1, le=10, description="Work stress level (1-10)")
    financial_stress: float = Field(..., ge=1, le=10, description="Financial stress level (1-10)")
    mood_rating: float = Field(..., ge=1, le=10, description="Mood rating (1-10)")
    energy_level: float = Field(..., ge=1, le=10, description="Energy level (1-10)")
    concentration_difficulty: float = Field(..., ge=1, le=10, description="Concentration difficulty (1-10)")


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    individuals: List[IndividualFeatures] = Field(..., min_items=1, max_items=1000)
    include_metadata: bool = Field(default=True, description="Include prediction metadata")


class PredictionResult(BaseModel):
    """Result for a single prediction."""
    depression: Dict[str, Union[str, float, bool]] = Field(..., description="Depression prediction results")
    anxiety: Dict[str, Union[str, float, bool]] = Field(..., description="Anxiety prediction results")


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

    @validator('model_type')
    def validate_model_type(cls, v):
        if v not in ['random_forest', 'logistic_regression']:
            raise ValueError('model_type must be either "random_forest" or "logistic_regression"')
        return v


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_info: Optional[Dict[str, str]] = Field(None, description="Model metadata")


# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # Startup
    logger.info("Starting Mental Wellness API...")
    await load_default_model()
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
    global predictor, data_processor, feature_engineer, model_metadata
    
    try:
        # Initialize components
        data_processor = DataProcessor()
        feature_engineer = FeatureEngineer()
        
        # Try to load existing model
        model_path = "/app/models/production_model.joblib"
        if os.path.exists(model_path):
            predictor = MentalWellnessPredictor()
            predictor.load_model(model_path)
            model_metadata = {
                "model_path": model_path,
                "loaded_at": datetime.now().isoformat(),
                "model_type": predictor.model_type,
                "version": "1.0.0"
            }
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.info("No pre-trained model found. Use /train endpoint to train a model.")
            
    except Exception as e:
        logger.error(f"Error during model loading: {e}")


def get_predictor():
    """Dependency to get the current predictor instance."""
    if predictor is None or not predictor.is_trained:
        raise HTTPException(
            status_code=503, 
            detail="Model not trained. Please train a model first using /train endpoint."
        )
    return predictor


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
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        model_loaded=predictor is not None and predictor.is_trained,
        model_info=model_metadata if model_metadata else None
    )


@app.post("/train")
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """Train a new mental wellness prediction model."""
    try:
        # Initialize components
        global predictor, data_processor, feature_engineer, model_metadata
        
        data_processor = DataProcessor()
        feature_engineer = FeatureEngineer()
        predictor = MentalWellnessPredictor(model_type=request.model_type)
        
        # Load or generate data
        if request.use_sample_data:
            df = data_processor._generate_sample_data(request.sample_size)
            logger.info(f"Generated {len(df)} samples of synthetic data")
        else:
            raise HTTPException(
                status_code=400,
                detail="Custom data training not implemented. Use use_sample_data=true"
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
        
        # Save model
        model_path = "/app/models/api_trained_model.joblib"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        predictor.save_model(model_path)
        
        # Update metadata
        model_metadata = {
            "model_path": model_path,
            "trained_at": datetime.now().isoformat(),
            "model_type": request.model_type,
            "test_size": request.test_size,
            "sample_size": request.sample_size,
            "version": "1.0.0"
        }
        
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
            anxiety=results['anxiety']
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
                anxiety=results['anxiety']
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
    if not model_metadata:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    return {
        "model_metadata": model_metadata,
        "is_trained": predictor is not None and predictor.is_trained,
        "feature_columns": predictor.feature_columns if predictor else []
    }


@app.delete("/model")
async def unload_model():
    """Unload the current model."""
    global predictor, model_metadata
    
    predictor = None
    model_metadata = {}
    
    return {"message": "Model unloaded successfully"}


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
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )