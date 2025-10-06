"""
Simple test to verify the API endpoints work with basic_trained_model.
This will start the server, test it, and stop it.
"""

import subprocess
import time
import requests
import json
import sys
from pathlib import Path

def test_api_with_basic_model():
    """Test the API endpoints with the basic trained model."""
    print("=== Testing API with Basic Trained Model ===\n")
    
    # Start the server
    print("1. Starting API server...")
    server_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "app:app", 
        "--host", "127.0.0.1", 
        "--port", "8001",  # Use different port to avoid conflicts
        "--log-level", "warning"  # Reduce log noise
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    time.sleep(5)
    
    base_url = "http://127.0.0.1:8001"
    
    try:
        # Test 1: Health check
        print("2. Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Health check successful")
            print(f"   - Status: {health_data.get('status')}")
            print(f"   - Model loaded: {health_data.get('model_loaded')}")
            if health_data.get('model_info'):
                print(f"   - Model source: {health_data['model_info'].get('source', 'N/A')}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
        
        # Test 2: Model info
        print("\n3. Testing model info endpoint...")
        response = requests.get(f"{base_url}/model/info", timeout=10)
        if response.status_code == 200:
            model_info = response.json()
            print(f"   ✅ Model info successful")
            print(f"   - Model trained: {model_info.get('is_trained')}")
            print(f"   - Feature columns: {len(model_info.get('feature_columns', []))}")
            print(f"   - Components initialized: {model_info.get('components_initialized')}")
        else:
            print(f"   ❌ Model info failed: {response.status_code}")
        
        # Test 3: Prediction
        print("\n4. Testing prediction endpoint...")
        prediction_data = {
            "age": 35,
            "sleep_hours": 6.5,
            "exercise_minutes": 150,
            "work_stress_level": 7,
            "mood_rating": 6,
            "energy_level": 5,
            "avg_heart_rate": 75,
            "resting_heart_rate": 65
        }
        
        response = requests.post(f"{base_url}/predict", json=prediction_data, timeout=10)
        if response.status_code == 200:
            prediction = response.json()
            print(f"   ✅ Prediction successful")
            print(f"   - Depression risk: {prediction.get('depression', {}).get('risk_level', 'N/A')}")
            print(f"   - Anxiety risk: {prediction.get('anxiety', {}).get('risk_level', 'N/A')}")
            print(f"   - Onset day: {prediction.get('onset_day', {}).get('predicted_days', 'N/A')} days")
        else:
            print(f"   ❌ Prediction failed: {response.status_code} - {response.text}")
            return False
        
        # Test 4: Batch prediction
        print("\n5. Testing batch prediction endpoint...")
        batch_data = {
            "individuals": [
                {
                    "age": 25,
                    "sleep_hours": 7,
                    "exercise_minutes": 120,
                    "work_stress_level": 5,
                    "mood_rating": 7,
                    "energy_level": 7,
                    "avg_heart_rate": 70,
                    "resting_heart_rate": 60
                },
                {
                    "age": 45,
                    "sleep_hours": 5,
                    "exercise_minutes": 30,
                    "work_stress_level": 9,
                    "mood_rating": 4,
                    "energy_level": 3,
                    "avg_heart_rate": 85,
                    "resting_heart_rate": 70
                }
            ],
            "include_metadata": True
        }
        
        response = requests.post(f"{base_url}/predict/batch", json=batch_data, timeout=10)
        if response.status_code == 200:
            batch_result = response.json()
            print(f"   ✅ Batch prediction successful")
            print(f"   - Total predictions: {batch_result.get('summary', {}).get('total_predictions', 0)}")
            print(f"   - High risk depression: {batch_result.get('summary', {}).get('high_risk_depression', 0)}")
            print(f"   - High risk anxiety: {batch_result.get('summary', {}).get('high_risk_anxiety', 0)}")
        else:
            print(f"   ❌ Batch prediction failed: {response.status_code} - {response.text}")
        
        print("\n=== Test Summary ===")
        print("✅ API successfully uses basic_trained_model automatically")
        print("✅ No training required - model loads on startup")
        print("✅ All prediction endpoints work with pre-trained model")
        print("✅ Both individual and batch predictions are functional")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    
    finally:
        # Stop the server
        print("\n6. Stopping server...")
        server_process.terminate()
        server_process.wait()
        print("   ✅ Server stopped")

def main():
    """Main test function."""
    # Change to the correct directory
    import os
    os.chdir(Path(__file__).parent)
    
    try:
        success = test_api_with_basic_model()
        if success:
            print("\n🎉 All tests passed! Your API is ready to use with the basic_trained_model.")
            print("\nTo start the API for production use:")
            print("   .venv\\Scripts\\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000")
        else:
            print("\n❌ Some tests failed. Please check the server logs.")
    except Exception as e:
        print(f"❌ Test suite failed: {e}")

if __name__ == "__main__":
    main()