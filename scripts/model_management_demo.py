#!/usr/bin/env python3
"""
Model Management Demo Script

This script demonstrates the model management capabilities of the Mental Wellness API.
It shows how to:
1. Train multiple models
2. List available models
3. Promote models to production
4. Make predictions with different models

Usage:
    python scripts/model_management_demo.py
"""

import requests
import json
import time
from datetime import datetime

# API base URL
API_BASE = "http://localhost:8000"

def make_request(method, endpoint, data=None):
    """Make an API request with error handling."""
    url = f"{API_BASE}{endpoint}"
    try:
        if method.upper() == "GET":
            response = requests.get(url)
        elif method.upper() == "POST":
            response = requests.post(url, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request to {url}: {e}")
        return None

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_json(data, title="Response"):
    """Print JSON data in a formatted way."""
    print(f"\n{title}:")
    print(json.dumps(data, indent=2))

def check_api_health():
    """Check if the API is healthy."""
    print_section("Checking API Health")
    response = make_request("GET", "/health")
    if response:
        print_json(response, "Health Status")
        return True
    else:
        print("❌ API is not available. Make sure the Docker container is running.")
        return False

def train_multiple_models():
    """Train multiple models with different configurations."""
    print_section("Training Multiple Models")
    
    models_to_train = [
        {
            "model_type": "random_forest",
            "sample_size": 1500,
            "test_size": 0.2,
            "name": "Random Forest Model"
        },
        {
            "model_type": "logistic_regression", 
            "sample_size": 2000,
            "test_size": 0.25,
            "name": "Logistic Regression Model"
        },
        {
            "model_type": "gradient_boosting",
            "sample_size": 1800,
            "test_size": 0.2,
            "name": "Gradient Boosting Model"
        }
    ]
    
    trained_models = []
    
    for i, config in enumerate(models_to_train, 1):
        print(f"\n🚀 Training Model {i}: {config['name']}")
        
        response = make_request("POST", "/train", config)
        if response:
            print(f"✅ {config['name']} trained successfully")
            print(f"   Model Path: {response['model_metadata']['model_path']}")
            print(f"   Training Samples: {response['model_metadata']['training_samples']}")
            trained_models.append(response['model_metadata'])
        else:
            print(f"❌ Failed to train {config['name']}")
        
        # Small delay between training sessions
        time.sleep(2)
    
    return trained_models

def list_available_models():
    """List all available models."""
    print_section("Available Models")
    
    response = make_request("GET", "/models")
    if response:
        print(f"📁 Model Directory: {response['model_directory']}")
        print(f"🏆 Production Model: {response.get('production_model', 'None')}")
        print(f"📊 Total Models: {len(response['available_models'])}")
        
        print("\nModel Details:")
        for model_id, details in response['available_models'].items():
            status = "🏆 PRODUCTION" if details.get('is_production', False) else "🧪 EXPERIMENTAL"
            print(f"  {status} {model_id}")
            print(f"    Created: {details['created_at']}")
            print(f"    Path: {details['model_path']}")
            if 'performance_metrics' in details:
                metrics = details['performance_metrics']
                if 'depression' in metrics:
                    print(f"    Depression Accuracy: {metrics['depression'].get('test_accuracy', 'N/A')}")
                if 'anxiety' in metrics:
                    print(f"    Anxiety Accuracy: {metrics['anxiety'].get('test_accuracy', 'N/A')}")
            print()
        
        return response['available_models']
    else:
        print("❌ Failed to retrieve model list")
        return {}

def get_current_model_info():
    """Get information about the currently loaded model."""
    print_section("Current Model Information")
    
    response = make_request("GET", "/model/info")
    if response:
        metadata = response['model_metadata']
        print(f"📍 Current Model: {metadata.get('model_path', 'Unknown')}")
        print(f"🔧 Model Type: {metadata.get('model_type', 'Unknown')}")
        print(f"📅 Trained At: {metadata.get('trained_at', 'Unknown')}")
        print(f"📊 Training Samples: {metadata.get('training_samples', 'Unknown')}")
        print(f"🏷️  Version: {metadata.get('version', 'Unknown')}")
        
        if 'performance_metrics' in metadata:
            print("\n📈 Performance Metrics:")
            metrics = metadata['performance_metrics']
            for condition, scores in metrics.items():
                if isinstance(scores, dict):
                    print(f"  {condition.title()}:")
                    for metric, value in scores.items():
                        print(f"    {metric}: {value}")
        
        return response
    else:
        print("❌ No model currently loaded")
        return None

def test_predictions():
    """Test predictions with the current model."""
    print_section("Testing Predictions")
    
    test_cases = [
        {
            "name": "Low Risk Individual",
            "features": {
                "age": 25,
                "sleep_hours": 8.0,
                "exercise_minutes": 150,
                "avg_heart_rate": 9,
                "work_stress_level": 3,
                "mood_score": 8,
                "fitness_level": 9,
                "resting_heart_rate": 58
            }
        },
        {
            "name": "High Risk Individual",
            "features": {
                "age": 45,
                "sleep_hours": 4.5,
                "exercise_minutes": 0,
                "avg_heart_rate": 2,
                "work_stress_level": 9,
                "mood_score": 3,
                "fitness_level": 2,
                "resting_heart_rate": 85
            }
        },
        {
            "name": "Moderate Risk Individual",
            "features": {
                "age": 35,
                "sleep_hours": 6.5,
                "exercise_minutes": 60,
                "avg_heart_rate": 6,
                "work_stress_level": 6,
                "mood_score": 5,
                "fitness_level": 5,
                "resting_heart_rate": 70
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case['name']}")
        response = make_request("POST", "/predict", test_case['features'])
        
        if response:
            print(f"  Depression: {response['depression']['risk_level']} "
                  f"(Probability: {response['depression']['probability']:.3f})")
            print(f"  Anxiety: {response['anxiety']['risk_level']} "
                  f"(Probability: {response['anxiety']['probability']:.3f})")
        else:
            print(f"  ❌ Prediction failed for {test_case['name']}")

def promote_best_model(available_models):
    """Promote the model with the best performance to production."""
    print_section("Model Promotion")
    
    if not available_models:
        print("❌ No models available for promotion")
        return
    
    # Find the model with the best average accuracy (simple heuristic)
    best_model = None
    best_score = 0
    
    for model_id, details in available_models.items():
        if details.get('is_production', False):
            continue  # Skip current production model
            
        if 'performance_metrics' in details:
            metrics = details['performance_metrics']
            dep_acc = metrics.get('depression', {}).get('test_accuracy', 0)
            anx_acc = metrics.get('anxiety', {}).get('test_accuracy', 0)
            avg_score = (dep_acc + anx_acc) / 2
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = model_id
    
    if not best_model:
        print("❌ No suitable model found for promotion")
        return
    
    print(f"🏆 Promoting best performing model: {best_model}")
    print(f"   Average Accuracy: {best_score:.3f}")
    
    # Extract model name and version from ID
    parts = best_model.split('_')
    if len(parts) >= 2:
        version = parts[-1]
        model_name = '_'.join(parts[:-1])
        
        response = make_request("POST", f"/models/{model_name}/{version}/promote")
        if response:
            print(f"✅ {response['message']}")
        else:
            print(f"❌ Failed to promote model {best_model}")
    else:
        print(f"❌ Could not parse model ID: {best_model}")

def main():
    """Main demo function."""
    print("🤖 Mental Wellness Model Management Demo")
    print("=" * 60)
    print("This demo will showcase the model management capabilities.")
    print("Make sure the API server is running at http://localhost:8000")
    
    # Check API health
    if not check_api_health():
        return
    
    # Show initial state
    get_current_model_info()
    
    # Train multiple models
    trained_models = train_multiple_models()
    
    # List all available models
    available_models = list_available_models()
    
    # Test predictions with current model
    test_predictions()
    
    # Promote the best performing model
    if available_models:
        promote_best_model(available_models)
        
        # Show updated model info
        print_section("Updated Model Information")
        get_current_model_info()
        
        # Test predictions with the new production model
        test_predictions()
    
    print_section("Demo Complete")
    print("🎉 Model management demo completed successfully!")
    print("\nYou can now:")
    print("• Access models from the host system in the ./models/ directory")
    print("• Use the API endpoints to manage your models")
    print("• Deploy the API in production with persistent model storage")

if __name__ == "__main__":
    main()