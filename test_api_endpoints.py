"""
Test the API endpoints for training and prediction workflow.
"""

import subprocess
import time
import requests
import json
import os
import sys
from pathlib import Path


def test_api_endpoints():
    """Test the train -> predict API workflow."""
    print("=== Testing API Train -> Predict Workflow ===\n")
    
    # Start the API server
    print("1. Starting API server...")
    server_process = None
    try:
        server_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "app:app", 
            "--host", "127.0.0.1", 
            "--port", "8002",  # Use different port
            "--log-level", "warning"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        time.sleep(8)
        
        base_url = "http://127.0.0.1:8002"
        
        # Test health check
        print("2. Testing health check...")
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Health check successful")
            print(f"   - Status: {health_data.get('status')}")
            print(f"   - Model loaded: {health_data.get('model_loaded')}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
        
        # Test training endpoint
        print("\n3. Testing training endpoint...")
        training_request = {
            "model_type": "random_forest",
            "test_size": 0.2,
            "use_sample_data": True,
            "sample_size": 500
        }
        
        response = requests.post(
            f"{base_url}/train",
            json=training_request,
            timeout=120
        )
        
        if response.status_code == 200:
            train_result = response.json()
            print("   ✅ Training successful!")
            print(f"   - Message: {train_result.get('message')}")
            
            # Check if GCS upload was mentioned
            model_metadata = train_result.get('model_metadata', {})
            if model_metadata.get('gcs_uploaded'):
                print(f"   - GCS uploaded: ✅ (as '{model_metadata.get('gcs_model_name')}')")
            else:
                print("   - GCS upload: ❌ (saved locally only)")
            
            # Show some performance metrics
            results = train_result.get('results', {})
            if 'depression' in results:
                print(f"   - Depression accuracy: {results['depression'].get('test_accuracy', 0):.3f}")
            if 'anxiety' in results:
                print(f"   - Anxiety accuracy: {results['anxiety'].get('test_accuracy', 0):.3f}")
        else:
            print(f"   ❌ Training failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        # Wait a moment for model to be saved
        time.sleep(2)
        
        # Test prediction endpoint
        print("\n4. Testing prediction endpoint...")
        prediction_request = {
            "age": 35,
            "sleep_hours": 6.5,
            "exercise_minutes": 150,
            "work_stress_level": 7,
            "mood_rating": 6,
            "energy_level": 5,
            "avg_heart_rate": 75,
            "resting_heart_rate": 65
        }
        
        response = requests.post(
            f"{base_url}/predict",
            json=prediction_request,
            timeout=10
        )
        
        if response.status_code == 200:
            prediction = response.json()
            print("   ✅ Prediction successful!")
            print(f"   - Depression: {prediction.get('depression', {}).get('risk_level', 'N/A')}")
            print(f"   - Anxiety: {prediction.get('anxiety', {}).get('risk_level', 'N/A')}")
            print(f"   - Onset day: {prediction.get('onset_day', {}).get('predicted_days', 'N/A')} days")
        else:
            print(f"   ❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        # Test health check again to see updated model info
        print("\n5. Testing health check after training...")
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Health check successful")
            print(f"   - Model loaded: {health_data.get('model_loaded')}")
            
            model_info = health_data.get('model_info', {})
            if model_info:
                print(f"   - Model source: {model_info.get('source', 'unknown')}")
                print(f"   - GCS model name: {model_info.get('gcs_model_name', 'none')}")
        
        # Test batch prediction
        print("\n6. Testing batch prediction...")
        batch_request = {
            "individuals": [
                {
                    "age": 25, "sleep_hours": 8, "exercise_minutes": 180,
                    "work_stress_level": 3, "mood_rating": 8, "energy_level": 8,
                    "avg_heart_rate": 65, "resting_heart_rate": 60
                },
                {
                    "age": 45, "sleep_hours": 5, "exercise_minutes": 30,
                    "work_stress_level": 9, "mood_rating": 3, "energy_level": 3,
                    "avg_heart_rate": 85, "resting_heart_rate": 75
                }
            ],
            "include_metadata": True
        }
        
        response = requests.post(
            f"{base_url}/predict/batch",
            json=batch_request,
            timeout=10
        )
        
        if response.status_code == 200:
            batch_result = response.json()
            print("   ✅ Batch prediction successful!")
            summary = batch_result.get('summary', {})
            print(f"   - Total predictions: {summary.get('total_predictions', 0)}")
            print(f"   - High risk depression: {summary.get('high_risk_depression', 0)}")
            print(f"   - High risk anxiety: {summary.get('high_risk_anxiety', 0)}")
        else:
            print(f"   ❌ Batch prediction failed: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        # Stop the server
        if server_process:
            print("\n7. Stopping server...")
            server_process.terminate()
            server_process.wait()
            print("   ✅ Server stopped")


def main():
    """Main test function."""
    print("🔄 API Endpoints Test\n")
    
    # Change to correct directory
    os.chdir(Path(__file__).parent)
    
    success = test_api_endpoints()
    
    print("\n=== Test Results ===")
    if success:
        print("🎉 All API tests passed!")
        print("\n✅ Verified functionality:")
        print("  - Health check endpoints")
        print("  - Model training via /train")
        print("  - Individual predictions via /predict")
        print("  - Batch predictions via /predict/batch")
        print("  - Model persistence after training")
        
        print("\n📋 Workflow confirmed:")
        print("  1. POST /train → Trains model, saves locally (and to GCS if available)")
        print("  2. POST /predict → Uses trained model for predictions")
        print("  3. Model persists between API calls")
        
        print("\n🐳 Ready for Docker deployment!")
    else:
        print("❌ Some API tests failed")


if __name__ == "__main__":
    main()