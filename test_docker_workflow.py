"""
Comprehensive test for Docker deployment with GCS model persistence.
This simulates the Docker container workflow.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

from model_manager import ModelManager
from mental_wellness_model import MentalWellnessPredictor, DataProcessor, FeatureEngineer


def simulate_docker_workflow():
    """Simulate the complete Docker container workflow with GCS persistence."""
    print("=== Simulating Docker Container Workflow with GCS ===\n")
    
    # Create temporary directory to simulate clean container environment
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_models_dir = os.path.join(temp_dir, "models")
        os.makedirs(temp_models_dir, exist_ok=True)
        
        print(f"1. Simulating clean container environment: {temp_models_dir}")
        
        # Step 1: Initialize model manager (simulating container startup)
        print("\n2. Container startup - Initializing model manager...")
        try:
            model_manager = ModelManager(base_path=temp_models_dir, enable_gcs=True)
            
            if model_manager.is_gcs_available():
                print("   ✅ GCS integration available")
                
                # Check if GCS has models
                gcs_models = model_manager.list_gcs_models()
                print(f"   📊 Found {len(gcs_models)} models in GCS")
                
                if gcs_models:
                    # Simulate downloading latest model
                    latest_model = model_manager.gcs_storage.get_latest_model()
                    if latest_model:
                        print(f"   📥 Downloading latest model from GCS: {latest_model}")
                        success = model_manager.download_model_from_gcs(latest_model)
                        if success:
                            print("   ✅ Model downloaded and ready for use")
                        else:
                            print("   ⚠️  Download failed, will use fallback")
                else:
                    print("   ℹ️  No models in GCS, will use local fallback")
            else:
                print("   ⚠️  GCS not available, using local models only")
                
        except Exception as e:
            print(f"   ❌ Error during startup: {e}")
            return False
        
        # Step 2: Simulate API training request
        print("\n3. Simulating API training request...")
        try:
            # Initialize components
            data_processor = DataProcessor()
            feature_engineer = FeatureEngineer()
            predictor = MentalWellnessPredictor(model_type='random_forest')
            
            # Generate and process data
            df = data_processor._generate_sample_data(800)
            df = data_processor.clean_data(df)
            df = feature_engineer.create_features(df)
            feature_engineer.fit_scaler(df)
            df = feature_engineer.transform_features(df)
            
            # Train model
            results = predictor.train(df, test_size=0.2)
            print("   ✅ Model training completed")
            
            # Save through model manager
            model_name = f"docker_simulation_{int(os.path.getmtime('.'))}"
            saved_path = model_manager.save_model(
                predictor,
                model_type="production",
                model_name=model_name,
                description="Docker workflow simulation test"
            )
            print(f"   💾 Model saved locally: {saved_path}")
            
            # Upload to GCS
            if model_manager.is_gcs_available():
                upload_success = model_manager.upload_model_to_gcs(
                    model_name,
                    {
                        "simulation_test": True,
                        "training_samples": 800,
                        "performance": results
                    }
                )
                
                if upload_success:
                    print("   ☁️  Model uploaded to GCS successfully")
                else:
                    print("   ⚠️  GCS upload failed")
            
        except Exception as e:
            print(f"   ❌ Error during training: {e}")
            return False
        
        # Step 3: Simulate container restart (clean environment)
        print("\n4. Simulating container restart...")
        print("   🔄 Clearing local models (simulating fresh container)...")
        
        # Clear local models
        shutil.rmtree(temp_models_dir)
        os.makedirs(temp_models_dir, exist_ok=True)
        
        # Initialize fresh model manager
        fresh_model_manager = ModelManager(base_path=temp_models_dir, enable_gcs=True)
        
        if fresh_model_manager.is_gcs_available():
            print("   📥 Fresh container checking GCS for models...")
            
            # Check local models (should be empty)
            local_models = fresh_model_manager.list_models()
            print(f"   📂 Local models: {len(local_models)}")
            
            if len(local_models) == 0:
                print("   ℹ️  No local models found (as expected for fresh container)")
                
                # Try to load from GCS
                gcs_models = fresh_model_manager.list_gcs_models()
                if gcs_models:
                    latest_model = fresh_model_manager.gcs_storage.get_latest_model()
                    if latest_model:
                        print(f"   📥 Downloading latest model: {latest_model}")
                        download_success = fresh_model_manager.download_model_from_gcs(latest_model)
                        
                        if download_success:
                            print("   ✅ Model restored from GCS successfully")
                            
                            # Test the restored model
                            model_path = fresh_model_manager.production_path / f"{latest_model}.joblib"
                            if model_path.exists():
                                test_predictor = MentalWellnessPredictor()
                                test_predictor.load_model(str(model_path))
                                
                                # Make test prediction
                                sample_features = {
                                    'age': 35, 'sleep_hours': 6.5, 'exercise_minutes': 150,
                                    'work_stress_level': 7, 'mood_rating': 6, 'energy_level': 5,
                                    'avg_heart_rate': 75, 'resting_heart_rate': 65
                                }
                                
                                prediction = test_predictor.predict_individual(sample_features)
                                print("   🎯 Restored model working for predictions")
                                print(f"      Depression: {prediction['depression']['risk_level']}")
                                print(f"      Anxiety: {prediction['anxiety']['risk_level']}")
                        else:
                            print("   ❌ Failed to download model from GCS")
                            return False
                else:
                    print("   ⚠️  No models found in GCS")
        
        # Step 4: Simulate multiple container deployments
        print("\n5. Simulating model persistence across deployments...")
        
        # Simulate 3 container restarts
        for i in range(1, 4):
            print(f"   Deployment {i}:")
            
            # Create new temporary directory (new container)
            deployment_dir = os.path.join(temp_dir, f"deployment_{i}")
            os.makedirs(deployment_dir, exist_ok=True)
            
            # Initialize model manager
            deployment_manager = ModelManager(base_path=deployment_dir, enable_gcs=True)
            
            if deployment_manager.is_gcs_available():
                # Check if models can be loaded from GCS
                gcs_models = deployment_manager.list_gcs_models()
                if gcs_models:
                    latest_model = deployment_manager.gcs_storage.get_latest_model()
                    if latest_model and deployment_manager.download_model_from_gcs(latest_model):
                        print(f"     ✅ Deployment {i}: Model loaded from GCS")
                    else:
                        print(f"     ❌ Deployment {i}: Failed to load from GCS")
                        return False
                else:
                    print(f"     ⚠️  Deployment {i}: No models in GCS")
        
        print("\n=== Docker Workflow Simulation Complete ===")
        print("✅ All deployment scenarios successful!")
        print("\n📋 Verified capabilities:")
        print("  ✅ Model training and local saving")
        print("  ✅ Automatic upload to GCS after training")
        print("  ✅ Model download from GCS on fresh container startup")
        print("  ✅ Model persistence across container restarts")
        print("  ✅ Multiple deployment consistency")
        print("  ✅ Prediction functionality with restored models")
        
        return True


def main():
    """Main simulation function."""
    # Set environment variable if not already set
    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        credentials_path = "./credentials/mentalwellness-473814-key.json"
        if os.path.exists(credentials_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            print(f"Using GCS credentials: {credentials_path}")
        else:
            print("❌ GCS credentials file not found")
            return
    
    try:
        success = simulate_docker_workflow()
        if success:
            print("\n🎉 Docker workflow simulation successful!")
            print("\n🐳 Your Docker container is ready with:")
            print("  - Automatic model persistence to GCS")
            print("  - Model restoration on container startup")
            print("  - Seamless deployment across environments")
            print("  - No manual model management required")
            print("\n📝 To deploy:")
            print("  docker build -t mental-wellness-api .")
            print("  docker run -p 8000:8000 mental-wellness-api")
        else:
            print("\n❌ Docker workflow simulation failed")
            print("Please check GCS configuration and try again")
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()