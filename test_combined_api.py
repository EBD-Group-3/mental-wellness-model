"""
Test script to verify the combined API functionality for training and prediction.
"""

import requests
import json
import time
from typing import Dict, Any


class APITester:
    """Test the Mental Wellness API endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health_check(self) -> Dict[str, Any]:
        """Test the health check endpoint."""
        print("Testing health check...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            result = response.json()
            print(f"Health check status: {response.status_code}")
            print(f"Model loaded: {result.get('model_loaded', False)}")
            return result
        except Exception as e:
            print(f"Health check failed: {e}")
            return {}
    
    def test_model_info(self) -> Dict[str, Any]:
        """Test the model info endpoint."""
        print("\nTesting model info...")
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            if response.status_code == 200:
                result = response.json()
                print(f"Model info status: {response.status_code}")
                print(f"Components initialized: {result.get('components_initialized', False)}")
                print(f"Model trained: {result.get('is_trained', False)}")
                return result
            else:
                print(f"Model info returned: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            print(f"Model info failed: {e}")
            return {}
    
    def test_training(self) -> Dict[str, Any]:
        """Test the model training endpoint."""
        print("\nTesting model training...")
        training_request = {
            "model_type": "random_forest",
            "test_size": 0.2,
            "use_sample_data": True,
            "sample_size": 500
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/train",
                json=training_request,
                timeout=120  # Training might take time
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Training status: {response.status_code}")
                print(f"Training completed successfully")
                
                # Print some metrics if available
                if "results" in result:
                    results = result["results"]
                    if "depression" in results:
                        print(f"Depression model accuracy: {results['depression'].get('test_accuracy', 'N/A'):.3f}")
                    if "anxiety" in results:
                        print(f"Anxiety model accuracy: {results['anxiety'].get('test_accuracy', 'N/A'):.3f}")
                return result
            else:
                print(f"Training failed: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            print(f"Training failed: {e}")
            return {}
    
    def test_prediction(self) -> Dict[str, Any]:
        """Test the prediction endpoint."""
        print("\nTesting individual prediction...")
        
        # Sample data for prediction
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
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=prediction_request
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Prediction status: {response.status_code}")
                print(f"Depression risk: {result.get('depression', {}).get('risk_level', 'N/A')}")
                print(f"Anxiety risk: {result.get('anxiety', {}).get('risk_level', 'N/A')}")
                print(f"Onset day prediction: {result.get('onset_day', {}).get('predicted_days', 'N/A')}")
                return result
            else:
                print(f"Prediction failed: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            print(f"Prediction failed: {e}")
            return {}
    
    def test_batch_prediction(self) -> Dict[str, Any]:
        """Test the batch prediction endpoint."""
        print("\nTesting batch prediction...")
        
        # Sample batch data
        batch_request = {
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
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json=batch_request
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Batch prediction status: {response.status_code}")
                print(f"Total predictions: {result.get('summary', {}).get('total_predictions', 0)}")
                print(f"High risk depression cases: {result.get('summary', {}).get('high_risk_depression', 0)}")
                print(f"High risk anxiety cases: {result.get('summary', {}).get('high_risk_anxiety', 0)}")
                return result
            else:
                print(f"Batch prediction failed: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            print(f"Batch prediction failed: {e}")
            return {}
    
    def run_full_test(self):
        """Run the complete test suite."""
        print("=== Mental Wellness API Test Suite ===\n")
        
        # Test 1: Health check
        health_result = self.test_health_check()
        
        # Test 2: Model info (before training)
        info_result = self.test_model_info()
        
        # Test 3: Training
        training_result = self.test_training()
        
        if training_result:
            # Wait a moment for training to complete
            time.sleep(2)
            
            # Test 4: Model info (after training)
            print("\nTesting model info after training...")
            info_after_training = self.test_model_info()
            
            # Test 5: Individual prediction
            prediction_result = self.test_prediction()
            
            # Test 6: Batch prediction
            batch_result = self.test_batch_prediction()
            
            print("\n=== Test Summary ===")
            print(f"Health check: {'✅ PASS' if health_result else '❌ FAIL'}")
            print(f"Model info: {'✅ PASS' if info_result else '❌ FAIL'}")
            print(f"Training: {'✅ PASS' if training_result else '❌ FAIL'}")
            print(f"Individual prediction: {'✅ PASS' if prediction_result else '❌ FAIL'}")
            print(f"Batch prediction: {'✅ PASS' if batch_result else '❌ FAIL'}")
        else:
            print("\n❌ Training failed - skipping prediction tests")


if __name__ == "__main__":
    # Create tester instance
    tester = APITester()
    
    # Run full test suite
    tester.run_full_test()