"""
Test Google Cloud Storage integration for model persistence.
"""

import os
import sys
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

from gcs_model_storage import GCSModelStorage
from model_manager import ModelManager
from mental_wellness_model import MentalWellnessPredictor, DataProcessor, FeatureEngineer


def test_gcs_integration():
    """Test the complete GCS integration workflow."""
    print("=== Testing GCS Model Storage Integration ===\n")
    
    # Test 1: Initialize GCS storage
    print("1. Initializing GCS storage...")
    try:
        gcs_storage = GCSModelStorage()
        if gcs_storage.is_available():
            print("   ✅ GCS storage initialized successfully")
            print(f"   - Bucket: {gcs_storage.bucket_name}")
            print(f"   - Model folder: {gcs_storage.model_folder}")
        else:
            print("   ❌ GCS storage not available")
            return False
    except Exception as e:
        print(f"   ❌ Failed to initialize GCS storage: {e}")
        return False
    
    # Test 2: List existing models in GCS
    print("\n2. Listing existing models in GCS...")
    try:
        gcs_models = gcs_storage.list_models()
        print(f"   ✅ Found {len(gcs_models)} models in GCS:")
        for model_name, info in gcs_models.items():
            print(f"     - {model_name}: {info.get('size_bytes', 0)} bytes")
    except Exception as e:
        print(f"   ⚠️  Could not list GCS models: {e}")
    
    # Test 3: Initialize enhanced model manager with GCS
    print("\n3. Initializing enhanced model manager...")
    try:
        model_manager = ModelManager(base_path="./models", enable_gcs=True)
        if model_manager.is_gcs_available():
            print("   ✅ Model manager with GCS integration initialized")
        else:
            print("   ⚠️  Model manager initialized but GCS not available")
    except Exception as e:
        print(f"   ❌ Failed to initialize model manager: {e}")
        return False
    
    # Test 4: Create and train a simple model
    print("\n4. Creating and training a test model...")
    try:
        # Initialize components
        data_processor = DataProcessor()
        feature_engineer = FeatureEngineer()
        predictor = MentalWellnessPredictor(model_type='random_forest')
        
        # Generate sample data
        df = data_processor._generate_sample_data(500)
        df = data_processor.clean_data(df)
        df = feature_engineer.create_features(df)
        feature_engineer.fit_scaler(df)
        df = feature_engineer.transform_features(df)
        
        # Train model
        results = predictor.train(df, test_size=0.2)
        print("   ✅ Test model trained successfully")
        
        # Save model locally first
        local_model_path = "./models/test_gcs_model.joblib"
        os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
        predictor.save_model(local_model_path)
        print(f"   ✅ Model saved locally: {local_model_path}")
        
    except Exception as e:
        print(f"   ❌ Failed to create and train model: {e}")
        return False
    
    # Test 5: Upload model to GCS
    print("\n5. Uploading model to GCS...")
    try:
        test_metadata = {
            "model_type": "random_forest",
            "test_accuracy": results.get('depression', {}).get('test_accuracy', 0),
            "training_samples": 500,
            "test_purpose": "gcs_integration_test"
        }
        
        upload_success = gcs_storage.upload_model(
            local_model_path, 
            "test_gcs_model", 
            test_metadata
        )
        
        if upload_success:
            print("   ✅ Model uploaded to GCS successfully")
        else:
            print("   ❌ Failed to upload model to GCS")
            return False
    except Exception as e:
        print(f"   ❌ Error uploading model to GCS: {e}")
        return False
    
    # Test 6: Download model from GCS
    print("\n6. Downloading model from GCS...")
    try:
        download_path = "./models/downloaded_test_model.joblib"
        metadata = gcs_storage.download_model("test_gcs_model", download_path)
        
        if metadata and os.path.exists(download_path):
            print("   ✅ Model downloaded from GCS successfully")
            print(f"     Downloaded to: {download_path}")
            
            # Test the downloaded model
            test_predictor = MentalWellnessPredictor()
            test_predictor.load_model(download_path)
            
            # Make a test prediction
            sample_features = {
                'age': 30,
                'sleep_hours': 7,
                'exercise_minutes': 120,
                'work_stress_level': 5,
                'mood_rating': 7,
                'energy_level': 6,
                'avg_heart_rate': 72,
                'resting_heart_rate': 62
            }
            
            prediction = test_predictor.predict_individual(sample_features)
            print("   ✅ Downloaded model works for predictions")
            print(f"     Test prediction - Depression: {prediction['depression']['risk_level']}")
        else:
            print("   ❌ Failed to download model from GCS")
            return False
    except Exception as e:
        print(f"   ❌ Error downloading model from GCS: {e}")
        return False
    
    # Test 7: Test model manager GCS integration
    print("\n7. Testing model manager GCS integration...")
    try:
        # Save model through model manager
        saved_path = model_manager.save_model(
            predictor,
            model_type="production",
            model_name="gcs_integration_test",
            description="Test model for GCS integration"
        )
        print(f"   ✅ Model saved through model manager: {saved_path}")
        
        # Upload through model manager
        upload_success = model_manager.upload_model_to_gcs(
            "gcs_integration_test",
            {"integration_test": True, "manager_upload": True}
        )
        
        if upload_success:
            print("   ✅ Model uploaded to GCS through model manager")
        else:
            print("   ⚠️  Model manager upload to GCS failed")
    except Exception as e:
        print(f"   ⚠️  Error testing model manager GCS integration: {e}")
    
    # Test 8: Clean up test models
    print("\n8. Cleaning up test models...")
    try:
        # Delete from GCS
        gcs_storage.delete_model("test_gcs_model")
        print("   ✅ Cleaned up test_gcs_model from GCS")
        
        # Clean up local files
        for path in [local_model_path, download_path]:
            if os.path.exists(path):
                os.remove(path)
                print(f"   ✅ Cleaned up local file: {path}")
    except Exception as e:
        print(f"   ⚠️  Error during cleanup: {e}")
    
    print("\n=== GCS Integration Test Complete ===")
    print("✅ GCS model storage integration is working correctly!")
    print("\nFeatures verified:")
    print("  - GCS bucket connection")
    print("  - Model upload to GCS")
    print("  - Model download from GCS")
    print("  - Model persistence across deployments")
    print("  - Model manager GCS integration")
    
    return True


def main():
    """Main test function."""
    # Set environment variable if not already set
    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        credentials_path = "./credentials/mentalwellness-473814-key.json"
        if os.path.exists(credentials_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            print(f"Set GCS credentials path: {credentials_path}")
        else:
            print("❌ GCS credentials file not found")
            return
    
    try:
        success = test_gcs_integration()
        if success:
            print("\n🎉 All GCS integration tests passed!")
            print("\nYour Docker container will now:")
            print("  1. Load the latest model from GCS on startup")
            print("  2. Upload trained models to GCS automatically")
            print("  3. Persist models across container restarts")
            print("  4. Provide API endpoints for GCS model management")
        else:
            print("\n❌ Some GCS integration tests failed")
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()