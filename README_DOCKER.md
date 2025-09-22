# Docker Usage Guide for Mental Wellness Model

This guide provides comprehensive examples for running the Mental Wellness Prediction Model CLI using Docker and Docker Compose.

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

## 📋 Available Commands

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
docker-compose run mental-wellness-cli python cli.py train --output /app/models/basic_model.pkl
```

### Training with Custom Configuration
Use a custom configuration file:
```bash
# First, place your config.yaml in the project root
docker-compose run mental-wellness-cli python cli.py train --config /app/custom_config.yaml --output /app/models/custom_model.pkl
```

### Training with Your Own Data
Train using your own CSV data file:
```bash
# Place your training_data.csv in the ./data directory
docker-compose run mental-wellness-cli python cli.py train --data /app/data/training_data.csv --output /app/models/trained_model.pkl
```

### Advanced Training Example
Complete training workflow with custom data and configuration:
```bash
docker-compose run mental-wellness-cli python cli.py train \
    --config /app/mental_wellness_model/config/default.yaml \
    --data /app/data/wellness_dataset.csv \
    --output /app/models/production_model.pkl
```

## 🔮 Prediction Examples

### Basic Prediction
Make a prediction using default parameters:
```bash
docker-compose run mental-wellness-cli python cli.py predict \
    --model /app/models/trained_model.pkl
```

### Detailed Individual Prediction
Predict mental wellness risk for a specific individual:
```bash
docker-compose run mental-wellness-cli python cli.py predict \
    --model /app/models/trained_model.pkl \
    --age 28 \
    --sleep-hours 6.5 \
    --exercise-frequency 3 \
    --social-interaction-score 7 \
    --work-stress-level 8 \
    --financial-stress 6 \
    --mood-rating 5 \
    --energy-level 4 \
    --concentration-difficulty 7
```

### High-Risk Individual Example
Example prediction for someone with concerning indicators:
```bash
docker-compose run mental-wellness-cli python cli.py predict \
    --model /app/models/trained_model.pkl \
    --age 35 \
    --sleep-hours 4.5 \
    --exercise-frequency 0 \
    --social-interaction-score 2 \
    --work-stress-level 9 \
    --financial-stress 8 \
    --mood-rating 3 \
    --energy-level 2 \
    --concentration-difficulty 9
```

### Low-Risk Individual Example
Example prediction for someone with positive indicators:
```bash
docker-compose run mental-wellness-cli python cli.py predict \
    --model /app/models/trained_model.pkl \
    --age 25 \
    --sleep-hours 8 \
    --exercise-frequency 5 \
    --social-interaction-score 9 \
    --work-stress-level 3 \
    --financial-stress 2 \
    --mood-rating 8 \
    --energy-level 9 \
    --concentration-difficulty 2
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
python cli.py train --output /app/models/dev_model.pkl

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
Your CSV training data should include these columns:
- `age`: Age of the individual (numeric)
- `sleep_hours`: Hours of sleep per night (numeric)
- `exercise_frequency`: Exercise sessions per week (numeric)
- `social_interaction_score`: Social interaction rating 1-10 (numeric)
- `work_stress_level`: Work stress level 1-10 (numeric)
- `financial_stress`: Financial stress level 1-10 (numeric)
- `mood_rating`: Mood rating 1-10 (numeric)
- `energy_level`: Energy level 1-10 (numeric)
- `concentration_difficulty`: Concentration difficulty 1-10 (numeric)

### Sample Data Directory Structure
```
./data/
├── training_data.csv          # Your training dataset
├── validation_data.csv        # Optional validation data
└── new_predictions.csv        # Data for batch predictions

./models/
├── production_model.pkl       # Trained production model
├── experimental_model.pkl     # Development models
└── backup_model.pkl          # Model backups

./output/
├── training_results.log       # Training logs
├── predictions.csv           # Batch prediction results
└── model_metrics.json       # Performance metrics
```

## 🔧 Configuration Examples

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
    --output /app/models/custom_model.pkl
```

## 🚀 Production Deployment Examples

### Complete Training Pipeline
```bash
# Step 1: Train the model with your data
docker-compose run mental-wellness-cli python cli.py train \
    --data /app/data/production_dataset.csv \
    --config /app/mental_wellness_model/config/default.yaml \
    --output /app/models/production_model.pkl

# Step 2: Validate the model with test cases
docker-compose run mental-wellness-cli python cli.py predict \
    --model /app/models/production_model.pkl \
    --age 30 --sleep-hours 7 --exercise-frequency 4

# Step 3: Use the model for predictions
docker-compose run mental-wellness-cli python cli.py predict \
    --model /app/models/production_model.pkl \
    --age 45 --sleep-hours 5.5 --exercise-frequency 1 \
    --social-interaction-score 4 --work-stress-level 8
```

### Batch Processing Workflow
For processing multiple individuals (you would need to modify the CLI to support batch input):
```bash
# Train once
docker-compose run mental-wellness-cli python cli.py train \
    --data /app/data/historical_data.csv \
    --output /app/models/batch_model.pkl

# Process multiple predictions (conceptual - would require CLI enhancement)
docker-compose run mental-wellness-cli python batch_predict.py \
    --model /app/models/batch_model.pkl \
    --input /app/data/individuals_to_assess.csv \
    --output /app/output/risk_assessments.csv
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
docker-compose run mental-wellness-cli python cli.py train --output /app/models/new_model.pkl
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