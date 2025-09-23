# Docker Usage Guide for Mental Wellness Model

This guide provides comprehensive examples for running the Mental Wellness Prediction Model using Docker and Docker Compose, including both CLI interface and REST API endpoints.

## 🐳 Quick Start

### Prerequisites
- Docker installed on your system
- Docker Compose installed (usually comes with Docker Desktop)

### Initial Setup
1. Clone the repository and navigate to the project directory
2. Create necessary directories for data persistence:
   ```bash
   mkdir -p data models output
   ```

3. Build the Docker image:
   ```bash
   docker-compose build
   ```

## 🚀 Running the API Server

### Production API with Gunicorn
Start the FastAPI server with Gunicorn for production:
```bash
docker-compose up mental-wellness-api
```

The API will be available at: http://localhost:8000

- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### Development API with Auto-reload
Start the development API server with auto-reload:
```bash
docker-compose up mental-wellness-api-dev
```

The development API will be available at: http://localhost:8001

## 📡 API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Train Model via API
```bash
curl -X POST "http://localhost:8000/train" \
     -H "Content-Type: application/json" \
     -d '{
       "model_type": "random_forest",
       "test_size": 0.2,
       "use_sample_data": true,
       "sample_size": 1000
     }'
```

### Single Prediction via API
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 28,
       "sleep_hours": 6.5,
       "exercise_minutes": 90,
       "avg_heart_rate": 7,
       "work_stress_level": 8,
       "mood_score": 5,
       "fitness_level": 4,
       "resting_heart_rate": 65
     }'
```

### Batch Predictions via API
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "individuals": [
         {
           "age": 28,
           "sleep_hours": 6.5,
           "exercise_minutes": 90,
           "avg_heart_rate": 7,
           "work_stress_level": 8,
           "mood_score": 5,
           "fitness_level": 4,
           "resting_heart_rate": 65
         },
         {
           "age": 35,
           "sleep_hours": 4.5,
           "exercise_minutes": 0,
           "avg_heart_rate": 2,
           "work_stress_level": 9,
           "mood_score": 3,
           "fitness_level": 2,
           "resting_heart_rate": 75
         }
       ],
       "include_metadata": true
     }'
```

### Get Model Information
```bash
curl http://localhost:8000/model/info
```

## � Service Management

### Start All Services
```bash
docker-compose up -d
```

### Start Specific Services
```bash
# Production API only
docker-compose up -d mental-wellness-api

# Development API only
docker-compose up -d mental-wellness-api-dev

# CLI service for interactive use
docker-compose up -d mental-wellness-cli
```

### View Service Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f mental-wellness-api
```

### Stop Services
```bash
# Stop all services
docker-compose down

# Stop specific service
docker-compose stop mental-wellness-api
```

### Scale Services
```bash
# Run multiple API instances behind a load balancer
docker-compose up -d --scale mental-wellness-api=3
```

## 📋 CLI Commands (Legacy Interface)

### Show CLI Help
```bash
docker-compose run mental-wellness-cli python cli.py --help
```

### Show Subcommand Help
```bash
# Help for training
docker-compose run mental-wellness-cli python cli.py train --help

# Help for prediction
docker-compose run mental-wellness-cli python cli.py predict --help
```

## 🎯 Training Examples

### Basic Training (Using Synthetic Data)
Train a model using the default synthetic data generator:
```bash
docker-compose run mental-wellness-cli python cli.py train --output /app/models/basic_model.joblib
```

### Training with Custom Configuration
Use a custom configuration file:
```bash
# First, place your config.yaml in the project root
docker-compose run mental-wellness-cli python cli.py train --config /app/custom_config.yaml --output /app/models/custom_model.joblib
```

### Training with Your Own Data
Train using your own CSV data file:
```bash
# Place your training_data.csv in the ./data directory
docker-compose run mental-wellness-cli python cli.py train --data /app/data/training_data.csv --output /app/models/trained_model.joblib
```

### Advanced Training Example
Complete training workflow with custom data and configuration:
```bash
docker-compose run mental-wellness-cli python cli.py train \
    --config /app/mental_wellness_model/config/default.yaml \
    --data /app/data/wellness_dataset.csv \
    --output /app/models/production_model.joblib
```

## 🔮 Prediction Examples

### Basic Prediction
Make a prediction using default parameters:
```bash
docker-compose run mental-wellness-cli python cli.py predict \
    --model /app/models/trained_model.joblib
```

### Detailed Individual Prediction
Predict mental wellness risk for a specific individual:
```bash
docker-compose run mental-wellness-cli python cli.py predict \
    --model /app/models/trained_model.joblib \
    --age 28 \
    --sleep-hours 6.5 \
    --exercise-frequency 3 \
    --social-interaction-score 7 \
    --work-stress-level 8 \
    --mood-score 5 \
    --fitness-level 4 \
    --smoking-status 0
```

### High-Risk Individual Example
Example prediction for someone with concerning indicators:
```bash
docker-compose run mental-wellness-cli python cli.py predict \
    --model /app/models/trained_model.joblib \
    --age 35 \
    --sleep-hours 4.5 \
    --exercise-frequency 0 \
    --social-interaction-score 2 \
    --work-stress-level 9 \
    --mood-score 3 \
    --fitness-level 2 \
    --smoking-status 1
```

### Low-Risk Individual Example
Example prediction for someone with positive indicators:
```bash
docker-compose run mental-wellness-cli python cli.py predict \
    --model /app/models/trained_model.joblib \
    --age 25 \
    --sleep-hours 8 \
    --exercise-frequency 5 \
    --social-interaction-score 9 \
    --work-stress-level 3 \
    --mood-score 8 \
    --fitness-level 9 \
    --smoking-status 0
```

### JSON-Based Predictions
For more structured input, you can create JSON files with prediction data:

#### Single Individual JSON Format
Create a file `./data/individual.json`:
```json
{
  "age": 28,
  "sleep_hours": 6.5,
  "exercise_minutes": 90,
  "avg_heart_rate": 7,
  "work_stress_level": 8,
  "mood_score": 5,
  "fitness_level": 4,
  "resting_heart_rate": 65
}
```

#### Multiple Individuals JSON Format
Create a file `./data/batch_predictions.json`:
```json
[
  {
    "id": "patient_001",
    "age": 28,
    "sleep_hours": 6.5,
    "exercise_minutes": 90,
    "avg_heart_rate": 7,
    "work_stress_level": 8,
    "mood_score": 5,
    "fitness_level": 4,
    "resting_heart_rate": 65
  },
  {
    "id": "patient_002",
    "age": 35,
    "sleep_hours": 4.5,
    "exercise_minutes": 0,
    "avg_heart_rate": 2,
    "work_stress_level": 9,
    "mood_score": 3,
    "fitness_level": 2,
    "resting_heart_rate": 75
  },
  {
    "id": "patient_003",
    "age": 25,
    "sleep_hours": 8,
    "exercise_minutes": 150,
    "avg_heart_rate": 9,
    "work_stress_level": 3,
    "mood_score": 8,
    "fitness_level": 9,
    "resting_heart_rate": 65
  }
]
```

#### Expected JSON Output Format
When using JSON input, the prediction output would be structured as:
```json
{
  "patient_001": {
    "depression": {
      "risk_level": "Moderate",
      "probability": 0.647,
      "prediction": true
    },
    "anxiety": {
      "risk_level": "High",
      "probability": 0.823,
      "prediction": true
    }
  },
  "patient_002": {
    "depression": {
      "risk_level": "Very High",
      "probability": 0.891,
      "prediction": true
    },
    "anxiety": {
      "risk_level": "Very High",
      "probability": 0.934,
      "prediction": true
    }
  },
  "patient_003": {
    "depression": {
      "risk_level": "Low",
      "probability": 0.156,
      "prediction": false
    },
    "anxiety": {
      "risk_level": "Low",
      "probability": 0.203,
      "prediction": false
    }
  }
}
```

*Note: JSON-based batch predictions would require CLI enhancement to support `--input-json` and `--output-json` parameters.*

## 🌐 API Response Examples

### Single Prediction Response
```json
{
  "depression": {
    "risk_level": "Moderate",
    "probability": 0.647,
    "prediction": true
  },
  "anxiety": {
    "risk_level": "High",
    "probability": 0.823,
    "prediction": true
  }
}
```

### Batch Prediction Response
```json
{
  "model_info": {
    "model_path": "/app/models/api_trained_model.joblib",
    "trained_at": "2025-09-22T10:30:00Z",
    "model_type": "random_forest",
    "version": "1.0.0"
  },
  "predictions": [
    {
      "depression": {
        "risk_level": "Moderate",
        "probability": 0.647,
        "prediction": true
      },
      "anxiety": {
        "risk_level": "High",
        "probability": 0.823,
        "prediction": true
      }
    },
    {
      "depression": {
        "risk_level": "Very High",
        "probability": 0.891,
        "prediction": true
      },
      "anxiety": {
        "risk_level": "Very High",
        "probability": 0.934,
        "prediction": true
      }
    }
  ],
  "summary": {
    "total_predictions": 2,
    "high_risk_depression": 1,
    "high_risk_anxiety": 2,
    "processing_time_ms": 45.2
  }
}
```

### Training Response
```json
{
  "message": "Model training completed successfully",
  "results": {
    "depression": {
      "test_accuracy": 0.845,
      "auc_score": 0.892,
      "cv_mean_accuracy": 0.838,
      "cv_std_accuracy": 0.023
    },
    "anxiety": {
      "test_accuracy": 0.823,
      "auc_score": 0.876,
      "cv_mean_accuracy": 0.819,
      "cv_std_accuracy": 0.031
    }
  },
  "model_metadata": {
    "model_path": "/app/models/api_trained_model.joblib",
    "trained_at": "2025-09-22T10:30:00Z",
    "model_type": "random_forest",
    "version": "1.0.0"
  }
}
```

### Health Check Response
```json
{
  "status": "healthy",
  "timestamp": "2025-09-22T10:30:00Z",
  "model_loaded": true,
  "model_info": {
    "model_path": "/app/models/production_model.joblib",
    "loaded_at": "2025-09-22T10:15:00Z",
    "model_type": "random_forest",
    "version": "1.0.0"
  }
}
```

## 🚀 Production Deployment with API

### Complete API Deployment
```bash
# Step 1: Build and start the API
docker-compose up -d mental-wellness-api

# Step 2: Train a model via API
curl -X POST "http://localhost:8000/train" \
     -H "Content-Type: application/json" \
     -d '{"model_type": "random_forest", "sample_size": 2000}'

# Step 3: Test with a prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"age": 30, "sleep_hours": 7, "exercise_minutes": 120, 
          "avg_heart_rate": 6, "work_stress_level": 5,
          "mood_score": 7, "fitness_level": 6,
          "resting_heart_rate": 65}'

# Step 4: Check API health
curl http://localhost:8000/health
```

### Load Balancing Multiple API Instances
```bash
# Start multiple API instances
docker-compose up -d --scale mental-wellness-api=3

# Use nginx or similar for load balancing
```

### API Monitoring
```bash
# Monitor API logs
docker-compose logs -f mental-wellness-api

# Check API metrics (if monitoring is set up)
curl http://localhost:8000/metrics
```

## 🛠️ Development Workflow

### Interactive Development Container
Start a development container for debugging and exploration:
```bash
docker-compose run mental-wellness-dev bash
```

Once inside the container, you can:
```bash
# Run Python interactively
python

# Execute the CLI directly
python cli.py train --output /app/models/dev_model.joblib

# Run tests
python -m pytest tests.py

# Explore the codebase
ls -la mental_wellness_model/
```

### Running the Example Script
Execute the provided example script:
```bash
docker-compose run mental-wellness-cli python example.py
```

### Running Tests
Execute the test suite:
```bash
docker-compose run mental-wellness-cli python tests.py
```

## 📊 Data Management

### Expected Data Format

#### CSV Format (for training data)
Your CSV training data should include these columns:
- `age`: Age of the individual (numeric)
- `sleep_hours`: Hours of sleep per night (numeric)
- `exercise_minutes`: Exercise minutes per week (numeric)
- `avg_heart_rate`: Average heart rate measurement 1-10 (numeric)
- `work_stress_level`: Work stress level 1-10 (numeric)
- `mood_score`: Mood score 1-10 (numeric)
- `fitness_level`: Fitness level 1-10 (numeric)
- `resting_heart_rate`: Resting heart rate (beats per minute) (numeric)

#### JSON Format (for prediction data)
For structured prediction input, use JSON format with the same field names:

**Single prediction JSON:**
```json
{
  "age": 30,
  "sleep_hours": 7.5,
  "exercise_minutes": 90,
  "avg_heart_rate": 6,
  "work_stress_level": 5,
  "mood_score": 7,
  "fitness_level": 6,
  "resting_heart_rate": 65
}
```

**Batch prediction JSON:**
```json
[
  {
    "id": "unique_identifier",
    "age": 30,
    "sleep_hours": 7.5,
    "exercise_minutes": 90,
    "avg_heart_rate": 6,
    "work_stress_level": 5,
    "mood_score": 7,
    "fitness_level": 6,
    "resting_heart_rate": 65
  }
]
```

### Sample Data Directory Structure
```
./data/
├── training_data.csv          # Your training dataset
├── validation_data.csv        # Optional validation data
├── new_predictions.csv        # Data for batch predictions
├── individual.json            # Single prediction JSON
├── batch_predictions.json     # Multiple predictions JSON
└── high_risk_cases.json      # Specific case studies

./models/
├── production_model.joblib       # Trained production model
├── experimental_model.joblib     # Development models
└── backup_model.joblib          # Model backups

./output/
├── training_results.log       # Training logs
├── predictions.csv           # Batch prediction results
├── predictions.json          # JSON prediction results
└── model_metrics.json       # Performance metrics
```

## 🔧 Model Management

The application now includes advanced model management features for production environments:

### Model Storage and Persistence
- Models are saved in the `/app/models` directory inside the container
- This directory is mounted as a Docker volume for persistence across container restarts
- Models are organized with versioning, metadata, and automated backup

### Model Versioning System
Each model has:
- **Name**: Descriptive identifier (e.g., `api_trained_random_forest_20241201_142345`)
- **Version**: Semantic versioning (e.g., `1.0.0`)
- **Metadata**: Training info, performance metrics, creation timestamp
- **Production Status**: Automatic promotion system

### Model Management API Endpoints

#### List All Available Models
Get an overview of all trained models:
```bash
curl -X GET http://localhost:8000/models
```

Example response:
```json
{
  "available_models": {
    "api_trained_random_forest_20241201_142345_1.0.0": {
      "model_path": "/app/models/api_trained_random_forest_20241201_142345_1.0.0.joblib",
      "metadata_path": "/app/models/api_trained_random_forest_20241201_142345_1.0.0.json",
      "created_at": "2024-12-01T14:23:45.123456",
      "is_production": true,
      "performance_metrics": {
        "depression": {"test_accuracy": 0.845, "auc_score": 0.892},
        "anxiety": {"test_accuracy": 0.823, "auc_score": 0.876}
      }
    },
    "experimental_model_20241201_120000_1.0.0": {
      "model_path": "/app/models/experimental_model_20241201_120000_1.0.0.joblib",
      "metadata_path": "/app/models/experimental_model_20241201_120000_1.0.0.json",
      "created_at": "2024-12-01T12:00:00.000000",
      "is_production": false
    }
  },
  "production_model": "api_trained_random_forest_20241201_142345_1.0.0",
  "model_directory": "/app/models"
}
```

#### Get Current Model Information
Check details about the currently loaded model:
```bash
curl -X GET http://localhost:8000/model/info
```

Example response:
```json
{
  "model_metadata": {
    "model_path": "/app/models/api_trained_random_forest_20241201_142345_1.0.0.joblib",
    "trained_at": "2024-12-01T14:23:45.123456",
    "model_type": "random_forest",
    "training_samples": 2000,
    "features": ["age", "sleep_hours", "exercise_minutes", ...],
    "performance_metrics": {
      "training_accuracy": 0.92,
      "depression": {"test_accuracy": 0.845},
      "anxiety": {"test_accuracy": 0.823}
    },
    "version": "1.0.0"
  },
  "is_trained": true,
  "feature_columns": ["age", "sleep_hours", "exercise_minutes", ...],
  "model_manager_info": {
    "available_models": {...},
    "production_model": "api_trained_random_forest_20241201_142345_1.0.0",
    "model_directory": "/app/models"
  }
}
```

#### Promote Model to Production
Switch the active production model:
```bash
curl -X POST http://localhost:8000/models/experimental_model_20241201_120000/1.0.0/promote
```

Example response:
```json
{
  "message": "Model experimental_model_20241201_120000 v1.0.0 promoted to production",
  "production_model_path": "/app/models/experimental_model_20241201_120000_1.0.0.joblib"
}
```

### Model Training with Versioning
When training via API, models are automatically versioned:
```bash
curl -X POST "http://localhost:8000/train" \
     -H "Content-Type: application/json" \
     -d '{
       "model_type": "random_forest",
       "sample_size": 2000,
       "test_size": 0.2
     }'
```

The response includes versioning information:
```json
{
  "message": "Model training completed successfully",
  "results": {...},
  "model_metadata": {
    "model_path": "/app/models/api_trained_random_forest_20241201_142345_1.0.0.joblib",
    "trained_at": "2024-12-01T14:23:45.123456",
    "model_type": "random_forest",
    "test_size": 0.2,
    "sample_size": 2000,
    "version": "1.0.0",
    "performance_metrics": {
      "depression": {"test_accuracy": 0.845, "auc_score": 0.892},
      "anxiety": {"test_accuracy": 0.823, "auc_score": 0.876}
    }
  }
}
```

### Accessing Models from Host System

Models are persistent and accessible from the host system:

```bash
# List all models from the host
ls -la models/

# View model metadata
cat models/api_trained_random_forest_20241201_142345_1.0.0.json

# Copy a model for backup
cp models/production_model.joblib /backup/location/

# Load a model in Python outside the container
python3 -c "
import joblib
model = joblib.load('models/api_trained_random_forest_20241201_142345_1.0.0.joblib')
print('Model loaded successfully:', type(model))
"
```

### Model Management Best Practices

1. **Regular Training**: Retrain models with new data periodically
2. **Version Control**: Always use semantic versioning for model releases
3. **Performance Monitoring**: Track model performance metrics over time
4. **Backup Strategy**: Keep backups of production models
5. **Promotion Testing**: Test models thoroughly before promoting to production

### Model Management Demo Script

We've included a comprehensive demo script that showcases all model management features:

```bash
# Start the API server first
docker-compose up -d mental-wellness-api

# Wait a few seconds for the API to start, then run the demo
docker-compose run mental-wellness-cli python scripts/model_management_demo.py
```

This demo script will:
1. ✅ Check API health and connectivity
2. 🚀 Train multiple models with different algorithms
3. 📋 List all available models with their metrics
4. 🧪 Test predictions with different risk profiles
5. 🏆 Promote the best performing model to production
6. 📊 Show updated model information and performance

The demo creates models using:
- Random Forest
- Logistic Regression  
- Gradient Boosting

And tests them with:
- Low risk individual (healthy lifestyle)
- High risk individual (multiple risk factors)
- Moderate risk individual (mixed indicators)

Example demo output:
```
🤖 Mental Wellness Model Management Demo
============================================================

🚀 Training Model 1: Random Forest Model
✅ Random Forest Model trained successfully
   Model Path: /app/models/api_trained_random_forest_20241201_142345_1.0.0.joblib
   Training Samples: 1500

🏆 Promoting best performing model: api_trained_gradient_boosting_20241201_143000_1.0.0
   Average Accuracy: 0.834

🧪 Testing: High Risk Individual
  Depression: Very High (Probability: 0.891)
  Anxiety: Very High (Probability: 0.934)
```

### Example Model Management Workflow

```bash
# Step 1: Train a new experimental model
curl -X POST "http://localhost:8000/train" \
     -H "Content-Type: application/json" \
     -d '{"model_type": "gradient_boosting", "sample_size": 3000}'

# Step 2: List all models to see the new one
curl -X GET http://localhost:8000/models

# Step 3: Test the new model with predictions
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"age": 30, "sleep_hours": 7, ...}'

# Step 4: If satisfied, promote to production
curl -X POST http://localhost:8000/models/api_trained_gradient_boosting_20241201_150000/1.0.0/promote

# Step 5: Verify the new production model is active
curl -X GET http://localhost:8000/model/info
```

### Custom Configuration File
Create a `custom_config.yaml`:
```yaml
# Custom Mental Wellness Model Configuration
model:
  type: "logistic_regression"  # Use logistic regression instead
  test_size: 0.25             # Larger test set
  random_state: 123

data:
  sample_size: 2000           # Generate more synthetic samples
  
features:
  scaling: true
  create_interactions: false  # Disable feature interactions

logging:
  level: "DEBUG"              # More verbose logging
```

### Running with Custom Config
```bash
docker-compose run mental-wellness-cli python cli.py train \
    --config /app/custom_config.yaml \
    --output /app/models/custom_model.joblib
```

## 🚀 Production Deployment Examples

### Complete Training Pipeline
```bash
# Step 1: Train the model with your data
docker-compose run mental-wellness-cli python cli.py train \
    --data /app/data/production_dataset.csv \
    --config /app/mental_wellness_model/config/default.yaml \
    --output /app/models/production_model.joblib

# Step 2: Validate the model with test cases
docker-compose run mental-wellness-cli python cli.py predict \
    --model /app/models/production_model.joblib \
    --age 30 --sleep-hours 7 --exercise-frequency 4

# Step 3: Use the model for predictions
docker-compose run mental-wellness-cli python cli.py predict \
    --model /app/models/production_model.joblib \
    --age 45 --sleep-hours 5.5 --exercise-frequency 1 \
    --social-interaction-score 4 --work-stress-level 8
```

### Batch Processing Workflow
For processing multiple individuals (you would need to modify the CLI to support batch input):

#### CSV-based Batch Processing
```bash
# Train once
docker-compose run mental-wellness-cli python cli.py train \
    --data /app/data/historical_data.csv \
    --output /app/models/batch_model.joblib

# Process multiple predictions (conceptual - would require CLI enhancement)
docker-compose run mental-wellness-cli python batch_predict.py \
    --model /app/models/batch_model.joblib \
    --input /app/data/individuals_to_assess.csv \
    --output /app/output/risk_assessments.csv
```

#### JSON-based Batch Processing
Enhanced CLI commands for JSON processing (conceptual implementation):
```bash
# Single JSON prediction
docker-compose run mental-wellness-cli python cli.py predict \
    --model /app/models/batch_model.joblib \
    --input-json /app/data/individual.json \
    --output-json /app/output/prediction_result.json

# Batch JSON predictions
docker-compose run mental-wellness-cli python cli.py predict \
    --model /app/models/batch_model.joblib \
    --input-json /app/data/batch_predictions.json \
    --output-json /app/output/batch_results.json \
    --batch-mode

# Streaming JSON predictions for large datasets
docker-compose run mental-wellness-cli python cli.py predict \
    --model /app/models/batch_model.joblib \
    --input-json /app/data/large_dataset.json \
    --output-json /app/output/streaming_results.json \
    --streaming \
    --batch-size 100
```

#### Expected JSON CLI Output
When using `--output-json`, the results would be saved in structured format:
```json
{
  "model_info": {
    "model_path": "/app/models/batch_model.joblib",
    "prediction_timestamp": "2025-09-22T10:30:00Z",
    "model_version": "0.1.0"
  },
  "predictions": {
    "patient_001": {
      "input": {
        "age": 28,
        "sleep_hours": 6.5,
        "exercise_minutes": 90,
        "avg_heart_rate": 7,
        "work_stress_level": 8,
        "mood_score": 5,
        "fitness_level": 4,
        "resting_heart_rate": 65
      },
      "results": {
        "depression": {
          "risk_level": "Moderate",
          "probability": 0.647,
          "prediction": true,
          "confidence": 0.834
        },
        "anxiety": {
          "risk_level": "High", 
          "probability": 0.823,
          "prediction": true,
          "confidence": 0.912
        }
      }
    }
  },
  "summary": {
    "total_predictions": 1,
    "high_risk_depression": 0,
    "high_risk_anxiety": 1,
    "processing_time_ms": 245
  }
}
```

## 🔍 Troubleshooting

### Common Issues and Solutions

#### Permission Issues
```bash
# If you encounter permission errors, try:
sudo docker-compose run mental-wellness-cli python cli.py --help
```

#### Memory Issues
```bash
# For large datasets, you might need to increase Docker memory limits
# Add to docker-compose.yml under the service:
# mem_limit: 4g
```

#### Model File Not Found
```bash
# Ensure the model file exists:
docker-compose run mental-wellness-cli ls -la /app/models/

# Train a model if none exists:
docker-compose run mental-wellness-cli python cli.py train --output /app/models/new_model.joblib
```

### Viewing Logs
```bash
# View container logs
docker-compose logs mental-wellness-cli

# Run with verbose logging
docker-compose run mental-wellness-cli python cli.py train --config /app/debug_config.yaml
```

## 📈 Model Performance Monitoring

### Checking Model Metrics
After training, the CLI outputs performance metrics:
- Test Accuracy
- AUC Score  
- Cross-validation mean accuracy
- Standard deviation

Example output:
```
DEPRESSION MODEL RESULTS:
  Test Accuracy: 0.845
  AUC Score: 0.892
  CV Mean Accuracy: 0.838 (+/- 0.023)

ANXIETY MODEL RESULTS:
  Test Accuracy: 0.823
  AUC Score: 0.876
  CV Mean Accuracy: 0.819 (+/- 0.031)
```

## 📝 Next Steps

1. **Prepare Your Data**: Format your data according to the expected schema
2. **Train Your Model**: Use your real data to train a production model
3. **Validate Performance**: Test the model with known cases
4. **Deploy**: Use the trained model for new predictions
5. **Monitor**: Track prediction accuracy over time

For more detailed information about the model architecture and features, see the main [README.md](./README.md) file.
