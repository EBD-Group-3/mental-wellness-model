#!/bin/bash

# Docker container startup script for Mental Wellness API
# Ensures models are available before starting the API

echo "=== Mental Wellness API Container Startup ==="

# Check if basic trained model exists
if [ -f "/app/models/basic_trained_model.joblib" ]; then
    echo "✅ Basic trained model found: /app/models/basic_trained_model.joblib"
    ls -la /app/models/basic_trained_model.joblib
else
    echo "⚠️  Basic trained model not found, checking alternative locations..."
    
    # Check if it exists in the source location and copy it
    if [ -f "/app/models/basic_trained_model.joblib" ]; then
        cp /app/models/basic_trained_model.joblib /app/models/
        echo "✅ Copied basic trained model to /app/models/"
    elif [ -f "./models/basic_trained_model.joblib" ]; then
        cp ./models/basic_trained_model.joblib /app/models/
        echo "✅ Copied basic trained model from ./models/ to /app/models/"
    else
        echo "❌ No basic trained model found. Creating a new one..."
        # Use Python to create a basic model if none exists
        python3 -c "
import sys
sys.path.insert(0, '/app')
from mental_wellness_model import MentalWellnessPredictor, DataProcessor, FeatureEngineer
import os

try:
    print('Creating basic model...')
    data_processor = DataProcessor()
    feature_engineer = FeatureEngineer()
    predictor = MentalWellnessPredictor()
    
    # Generate sample data
    df = data_processor._generate_sample_data(1000)
    df = data_processor.clean_data(df)
    df = feature_engineer.create_features(df)
    feature_engineer.fit_scaler(df)
    df = feature_engineer.transform_features(df)
    
    # Train and save
    predictor.train(df)
    predictor.save_model('/app/models/basic_trained_model.joblib')
    print('✅ Basic model created and saved')
except Exception as e:
    print(f'❌ Failed to create basic model: {e}')
    exit(1)
"
    fi
fi

# Check GCS credentials
if [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "✅ GCS credentials found: $GOOGLE_APPLICATION_CREDENTIALS"
else
    echo "⚠️  GCS credentials not found at: $GOOGLE_APPLICATION_CREDENTIALS"
fi

# List all available models
echo "📁 Available models in /app/models/:"
ls -la /app/models/ || echo "No models directory found"

# Test model loading before starting API
echo "🧪 Testing model loading..."
python3 -c "
import sys
sys.path.insert(0, '/app')
import os
from mental_wellness_model import MentalWellnessPredictor

model_paths = [
    '/app/models/basic_trained_model.joblib',
    '/app/models/api_trained_model.joblib',
    './models/basic_trained_model.joblib'
]

model_loaded = False
for path in model_paths:
    if os.path.exists(path):
        try:
            predictor = MentalWellnessPredictor()
            predictor.load_model(path)
            print(f'✅ Successfully loaded model from: {path}')
            print(f'   Model type: {getattr(predictor, \"model_type\", \"unknown\")}')
            print(f'   Is trained: {getattr(predictor, \"is_trained\", False)}')
            model_loaded = True
            break
        except Exception as e:
            print(f'❌ Failed to load model from {path}: {e}')

if not model_loaded:
    print('❌ No models could be loaded - API may not work properly')
    exit(1)
else:
    print('✅ Model loading test successful')
"

if [ $? -eq 0 ]; then
    echo "🚀 Starting Mental Wellness API..."
    # Start the application
    exec "$@"
else
    echo "❌ Startup checks failed"
    exit 1
fi