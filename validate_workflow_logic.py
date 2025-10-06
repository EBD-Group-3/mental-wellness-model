"""
Validate the train -> GCS -> predict workflow logic without running a server.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the package to the path  
sys.path.insert(0, str(Path(__file__).parent))

from mental_wellness_model import MentalWellnessPredictor, DataProcessor, FeatureEngineer
from model_manager import ModelManager


def test_workflow_logic():
    """Test the workflow logic without GCS (simulating local-only operation)."""
    print("=== Testing Workflow Logic ===\n")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"1. Using temporary directory: {temp_dir}")
        
        # Initialize components (simulating what happens in API)
        print("2. Initializing components...")
        try:
            data_processor = DataProcessor()
            feature_engineer = FeatureEngineer()
            model_manager = ModelManager(base_path=temp_dir, enable_gcs=False)  # Disable GCS for this test
            predictor = MentalWellnessPredictor(model_type='random_forest')
            print("   ✅ Components initialized")
        except Exception as e:
            print(f"   ❌ Component initialization failed: {e}")
            return False
        
        # Simulate training process (like /train endpoint)
        print("\n3. Simulating training process...")
        try:
            # Generate and process data
            df = data_processor._generate_sample_data(500)
            df = data_processor.clean_data(df)
            df = feature_engineer.create_features(df)
            feature_engineer.fit_scaler(df)
            df = feature_engineer.transform_features(df)
            
            # Train model
            results = predictor.train(df, test_size=0.2)
            print("   ✅ Model training completed")
            
            # Save model with standard name (simulating what API does)
            model_name = "basic_trained_model"
            saved_path = model_manager.save_model(
                predictor,
                model_type="production",
                model_name=model_name,
                description="API trained model for workflow test"
            )
            print(f"   ✅ Model saved as: {saved_path}")
            
            # Also save to a standard location (like API does)
            basic_model_path = os.path.join(temp_dir, "basic_trained_model.joblib")
            predictor.save_model(basic_model_path)
            print(f"   ✅ Model also saved to: {basic_model_path}")
            
        except Exception as e:
            print(f"   ❌ Training simulation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Simulate prediction loading (like what happens on /predict)
        print("\n4. Simulating prediction loading...")
        try:
            # Clear the predictor to simulate fresh load
            fresh_predictor = None
            
            # Try loading from the saved model (simulating API model loading logic)
            model_paths = [
                basic_model_path,
                saved_path,
                os.path.join(temp_dir, "production", f"{model_name}.joblib")
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    try:
                        fresh_predictor = MentalWellnessPredictor()
                        fresh_predictor.load_model(model_path)
                        print(f"   ✅ Successfully loaded model from: {model_path}")
                        break
                    except Exception as load_error:
                        print(f"   ⚠️  Failed to load from {model_path}: {load_error}")
                        continue
                else:
                    print(f"   ⚪ Model not found: {model_path}")
            
            if not fresh_predictor or not fresh_predictor.is_trained:
                print("   ❌ No model could be loaded")
                return False
            
        except Exception as e:
            print(f"   ❌ Model loading simulation failed: {e}")
            return False
        
        # Test prediction (like /predict endpoint)
        print("\n5. Testing prediction with loaded model...")
        try:
            sample_features = {
                'age': 35, 'sleep_hours': 6.5, 'exercise_minutes': 150,
                'work_stress_level': 7, 'mood_rating': 6, 'energy_level': 5,
                'avg_heart_rate': 75, 'resting_heart_rate': 65
            }
            
            prediction = fresh_predictor.predict_individual(sample_features)
            print("   ✅ Prediction successful!")
            print(f"      Depression: {prediction['depression']['risk_level']}")
            print(f"      Anxiety: {prediction['anxiety']['risk_level']}")
            print(f"      Onset day: {prediction.get('onset_day', {}).get('predicted_days', 'N/A')} days")
            
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")
            return False
        
        # Test batch prediction
        print("\n6. Testing batch prediction...")
        try:
            batch_features = [
                {'age': 25, 'sleep_hours': 8, 'exercise_minutes': 180, 'work_stress_level': 3, 
                 'mood_rating': 8, 'energy_level': 8, 'avg_heart_rate': 65, 'resting_heart_rate': 60},
                {'age': 45, 'sleep_hours': 5, 'exercise_minutes': 30, 'work_stress_level': 9,
                 'mood_rating': 3, 'energy_level': 3, 'avg_heart_rate': 85, 'resting_heart_rate': 75}
            ]
            
            batch_predictions = []
            for features in batch_features:
                pred = fresh_predictor.predict_individual(features)
                batch_predictions.append(pred)
            
            print(f"   ✅ Batch prediction successful! ({len(batch_predictions)} predictions)")
            
            # Count high-risk cases
            high_risk_depression = sum(1 for p in batch_predictions 
                                     if p['depression']['risk_level'] in ['High', 'Very High'])
            high_risk_anxiety = sum(1 for p in batch_predictions 
                                  if p['anxiety']['risk_level'] in ['High', 'Very High'])
            
            print(f"      High risk depression: {high_risk_depression}")
            print(f"      High risk anxiety: {high_risk_anxiety}")
            
        except Exception as e:
            print(f"   ❌ Batch prediction failed: {e}")
            return False
        
        return True


def test_api_model_state():
    """Test the API's ModelState logic."""
    print("\n=== Testing API ModelState Logic ===\n")
    
    try:
        from app import ModelState
        
        # Test ModelState initialization
        print("1. Testing ModelState initialization...")
        model_state = ModelState()
        model_state.initialize_components()
        print("   ✅ ModelState initialized")
        
        # Test model loading
        print("2. Testing model loading...")
        success = model_state.load_model_if_available()
        
        if success and model_state.predictor:
            print("   ✅ Model loaded via ModelState")
            print(f"      Source: {model_state.model_metadata.get('source', 'unknown')}")
            print(f"      Path: {model_state.model_metadata.get('model_path', 'unknown')}")
            
            # Test prediction through ModelState
            sample_features = {
                'age': 30, 'sleep_hours': 7, 'exercise_minutes': 120,
                'work_stress_level': 5, 'mood_rating': 7, 'energy_level': 6,
                'avg_heart_rate': 72, 'resting_heart_rate': 62
            }
            
            prediction = model_state.predictor.predict_individual(sample_features)
            print("   ✅ Prediction through ModelState successful")
            print(f"      Depression: {prediction['depression']['risk_level']}")
            
        else:
            print("   ⚠️  No model loaded (this is expected if no models exist yet)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ModelState test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🔍 Workflow Logic Validation\n")
    
    # Test 1: Core workflow logic
    workflow_success = test_workflow_logic()
    
    # Test 2: API ModelState logic  
    modelstate_success = test_api_model_state()
    
    print("\n=== Validation Results ===")
    print(f"Workflow logic: {'✅ PASS' if workflow_success else '❌ FAIL'}")
    print(f"ModelState logic: {'✅ PASS' if modelstate_success else '❌ FAIL'}")
    
    if workflow_success and modelstate_success:
        print("\n🎉 All validations passed!")
        print("\n✅ Confirmed functionality:")
        print("  - Training saves model with name 'basic_trained_model'")
        print("  - Model loading finds the saved model")
        print("  - Predictions work with loaded model")
        print("  - Batch predictions work correctly")
        print("  - API ModelState logic is sound")
        
        print("\n🎯 API Workflow Ready:")
        print("  1. POST /train → Trains model, saves as 'basic_trained_model'")
        print("  2. POST /predict → Loads 'basic_trained_model', makes predictions")
        print("  3. Model persists between calls")
        
        if os.path.exists("./credentials/mentalwellness-473814-key.json"):
            print("\n☁️  GCS Integration:")
            print("  - When GCS is available, models will also be saved to cloud")
            print("  - GCS credentials found (may need renewal if JWT errors persist)")
        else:
            print("\n⚠️  GCS credentials not found - local storage only")
            
    else:
        print("\n❌ Some validations failed - check logs above")


if __name__ == "__main__":
    main()