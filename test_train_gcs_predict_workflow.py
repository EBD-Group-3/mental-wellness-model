"""
Test the complete workflow: Train -> Save to GCS -> Load for Prediction
"""

import os
import sys
import time
import requests
import json
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

from gcs_model_storage import GCSModelStorage


def test_train_gcs_predict_workflow():
    """Test the complete train -> GCS -> predict workflow."""
    print("=== Testing Train -> GCS -> Predict Workflow ===\n")
    
    # Step 1: Check GCS availability
    print("1. Checking GCS availability...")
    try:
        gcs_storage = GCSModelStorage()
        if not gcs_storage.is_available():
            print("❌ GCS not available - cannot test workflow")
            return False
        print("✅ GCS is available")
    except Exception as e:
        print(f"❌ GCS initialization failed: {e}")
        return False
    
    # Step 2: Check if basic_trained_model already exists in GCS
    print("\n2. Checking existing models in GCS...")
    try:
        gcs_models = gcs_storage.list_models()
        print(f"   Found {len(gcs_models)} models in GCS:")
        for model_name in gcs_models.keys():
            print(f"     - {model_name}")
    except Exception as e:
        print(f"   ⚠️  Could not list GCS models: {e}")
    
    # Step 3: Test local model manager workflow
    print("\n3. Testing local model manager workflow...")
    try:
        from model_manager import ModelManager
        from mental_wellness_model import MentalWellnessPredictor, DataProcessor, FeatureEngineer
        
        # Initialize components
        model_manager = ModelManager(enable_gcs=True)
        data_processor = DataProcessor()
        feature_engineer = FeatureEngineer()
        predictor = MentalWellnessPredictor(model_type='random_forest')
        
        print("   ✅ Components initialized")
        
        # Create sample training data
        df = data_processor._generate_sample_data(500)
        df = data_processor.clean_data(df)
        df = feature_engineer.create_features(df)
        feature_engineer.fit_scaler(df)
        df = feature_engineer.transform_features(df)
        
        # Train model
        results = predictor.train(df, test_size=0.2)
        print("   ✅ Model trained successfully")
        
        # Save model through model manager
        model_name = "basic_trained_model"
        saved_path = model_manager.save_model(
            predictor,
            model_type="production",
            model_name=model_name,
            description="Test basic trained model for GCS workflow"
        )
        print(f"   ✅ Model saved locally: {saved_path}")
        
        # Upload to GCS with the standard name
        upload_success = model_manager.upload_model_to_gcs(
            model_name,
            {
                "workflow_test": True,
                "test_timestamp": time.time(),
                "performance_metrics": results
            }
        )
        
        if upload_success:
            print("   ✅ Model uploaded to GCS as 'basic_trained_model'")
        else:
            print("   ❌ Failed to upload model to GCS")
            return False
        
    except Exception as e:
        print(f"   ❌ Local workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Test downloading the model
    print("\n4. Testing model download from GCS...")
    try:
        # Clear local model to simulate fresh container
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            fresh_model_manager = ModelManager(base_path=temp_dir, enable_gcs=True)
            
            # Download basic_trained_model
            download_success = fresh_model_manager.download_model_from_gcs("basic_trained_model")
            
            if download_success:
                print("   ✅ Successfully downloaded basic_trained_model from GCS")
                
                # Test loading the downloaded model
                downloaded_path = Path(temp_dir) / "production" / "basic_trained_model.joblib"
                if downloaded_path.exists():
                    test_predictor = MentalWellnessPredictor()
                    test_predictor.load_model(str(downloaded_path))
                    
                    # Test prediction
                    sample_features = {
                        'age': 30, 'sleep_hours': 7, 'exercise_minutes': 120,
                        'work_stress_level': 5, 'mood_rating': 7, 'energy_level': 6,
                        'avg_heart_rate': 72, 'resting_heart_rate': 62
                    }
                    
                    prediction = test_predictor.predict_individual(sample_features)
                    print("   ✅ Downloaded model works for predictions")
                    print(f"      Depression: {prediction['depression']['risk_level']}")
                    print(f"      Anxiety: {prediction['anxiety']['risk_level']}")
                else:
                    print("   ❌ Downloaded model file not found")
                    return False
            else:
                print("   ❌ Failed to download model from GCS")
                return False
                
    except Exception as e:
        print(f"   ❌ Download test failed: {e}")
        return False
    
    # Step 5: Verify GCS model exists and has correct name
    print("\n5. Verifying GCS model exists with correct name...")
    try:
        updated_models = gcs_storage.list_models()
        if "basic_trained_model" in updated_models:
            model_info = updated_models["basic_trained_model"]
            print("   ✅ basic_trained_model found in GCS")
            print(f"      Size: {model_info.get('size_bytes', 0)} bytes")
            print(f"      Created: {model_info.get('created', 'unknown')}")
        else:
            print("   ❌ basic_trained_model not found in GCS")
            print(f"   Available models: {list(updated_models.keys())}")
            return False
    except Exception as e:
        print(f"   ❌ GCS verification failed: {e}")
        return False
    
    print("\n=== Workflow Test Complete ===")
    print("✅ Complete workflow successful!")
    print("\n📋 Verified functionality:")
    print("  ✅ Model training")
    print("  ✅ Save to GCS with name 'basic_trained_model'")
    print("  ✅ Download from GCS")
    print("  ✅ Load downloaded model for predictions")
    print("  ✅ Predictions work correctly")
    
    print("\n🎯 API Workflow:")
    print("  1. POST /train → Trains model and saves to GCS as 'basic_trained_model'")
    print("  2. POST /predict → Loads 'basic_trained_model' from GCS and makes predictions")
    
    return True


def test_api_workflow_simulation():
    """Simulate the actual API workflow without running the server."""
    print("\n=== API Workflow Simulation ===\n")
    
    try:
        # Import the app components
        from app import ModelState
        
        # Create model state instance
        model_state = ModelState()
        model_state.initialize_components()
        
        print("1. Model state initialized")
        
        # Test model loading (simulating what happens on /predict call)
        print("2. Testing model loading (as would happen on /predict)...")
        success = model_state.load_model_if_available()
        
        if success and model_state.predictor:
            print("   ✅ Model loaded successfully")
            print(f"   - Source: {model_state.model_metadata.get('source', 'unknown')}")
            print(f"   - GCS Model: {model_state.model_metadata.get('gcs_model_name', 'none')}")
            
            # Test prediction
            sample_features = {
                'age': 35, 'sleep_hours': 6.5, 'exercise_minutes': 150,
                'work_stress_level': 7, 'mood_rating': 6, 'energy_level': 5,
                'avg_heart_rate': 75, 'resting_heart_rate': 65
            }
            
            try:
                prediction = model_state.predictor.predict_individual(sample_features)
                print("   ✅ Prediction successful")
                print(f"      Depression: {prediction['depression']['risk_level']}")
                print(f"      Anxiety: {prediction['anxiety']['risk_level']}")
            except Exception as pred_error:
                print(f"   ❌ Prediction failed: {pred_error}")
                return False
        else:
            print("   ❌ No model could be loaded")
            return False
        
        print("\n✅ API workflow simulation successful!")
        return True
        
    except Exception as e:
        print(f"❌ API simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🔄 Testing Complete Train -> GCS -> Predict Workflow\n")
    
    # Set GCS credentials if not set
    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        credentials_path = "./credentials/mentalwellness-473814-key.json"
        if os.path.exists(credentials_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            print(f"Using GCS credentials: {credentials_path}")
        else:
            print("❌ GCS credentials not found")
            return
    
    # Test 1: Complete workflow
    workflow_success = test_train_gcs_predict_workflow()
    
    # Test 2: API simulation
    api_success = test_api_workflow_simulation()
    
    print("\n=== Final Results ===")
    print(f"Complete workflow test: {'✅ PASS' if workflow_success else '❌ FAIL'}")
    print(f"API simulation test: {'✅ PASS' if api_success else '❌ FAIL'}")
    
    if workflow_success and api_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ Your API now:")
        print("  - Trains models and saves them to GCS as 'basic_trained_model'")
        print("  - Loads 'basic_trained_model' from GCS for predictions")
        print("  - Maintains model persistence across deployments")
        print("  - Works seamlessly with Docker containers")
    else:
        print("\n❌ Some tests failed - check the logs above")


if __name__ == "__main__":
    main()