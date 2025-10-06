"""
Demonstration of the combined API and model functionality.
This script shows how the API integrates training and prediction capabilities.
"""

import sys
import os
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import our components
from mental_wellness_model import MentalWellnessPredictor, DataProcessor, FeatureEngineer
from model_manager import ModelManager


def demonstrate_combined_functionality():
    """Demonstrate the combined model and API functionality."""
    print("=== Mental Wellness Model Integration Demo ===\n")
    
    # Initialize components (same as API does)
    print("1. Initializing components...")
    data_processor = DataProcessor()
    feature_engineer = FeatureEngineer()
    model_manager = ModelManager()
    print("✅ Components initialized successfully")
    
    # Create predictor
    print("\n2. Creating predictor...")
    predictor = MentalWellnessPredictor(model_type='random_forest')
    print("✅ Predictor created")
    
    # Generate sample data for training
    print("\n3. Generating sample training data...")
    df = data_processor._generate_sample_data(1000)
    print(f"✅ Generated {len(df)} samples")
    
    # Process data (same as API does)
    print("\n4. Processing data...")
    df = data_processor.clean_data(df)
    if not data_processor.validate_data(df):
        print("❌ Data validation failed")
        return
    print("✅ Data cleaned and validated")
    
    # Feature engineering
    print("\n5. Performing feature engineering...")
    df = feature_engineer.create_features(df)
    feature_engineer.fit_scaler(df)
    df = feature_engineer.transform_features(df)
    print("✅ Features engineered")
    
    # Train model
    print("\n6. Training model...")
    results = predictor.train(df, test_size=0.2)
    print("✅ Model training completed")
    
    # Print training results
    print("\n=== Training Results ===")
    for condition, metrics in results.items():
        print(f"\n{condition.upper()} Model:")
        if condition in ['depression', 'anxiety']:
            print(f"  Test Accuracy: {metrics.get('test_accuracy', 0):.3f}")
            print(f"  AUC Score: {metrics.get('auc_score', 0):.3f}")
        else:  # onset_day
            print(f"  R² Score: {metrics.get('test_r2_score', 0):.3f}")
            print(f"  RMSE: {metrics.get('rmse', 0):.2f} days")
    
    # Save model
    print("\n7. Saving model...")
    model_path = "./models/demo_trained_model.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    predictor.save_model(model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Test predictions (same as API does)
    print("\n8. Testing predictions...")
    
    # Sample individual for prediction
    sample_features = {
        'age': 35,
        'sleep_hours': 6.5,
        'exercise_minutes': 150,
        'work_stress_level': 7,
        'mood_rating': 6,
        'energy_level': 5,
        'avg_heart_rate': 75,
        'resting_heart_rate': 65
    }
    
    # Make prediction
    prediction_result = predictor.predict_individual(sample_features)
    
    print("\n=== Prediction Results ===")
    print(f"Depression Risk: {prediction_result['depression']['risk_level']}")
    print(f"Depression Probability: {prediction_result['depression']['probability']:.3f}")
    print(f"Anxiety Risk: {prediction_result['anxiety']['risk_level']}")
    print(f"Anxiety Probability: {prediction_result['anxiety']['probability']:.3f}")
    print(f"Onset Day Prediction: {prediction_result['onset_day']['predicted_days']:.1f} days")
    
    # Test loading saved model
    print("\n9. Testing model loading...")
    new_predictor = MentalWellnessPredictor()
    new_predictor.load_model(model_path)
    print("✅ Model loaded successfully")
    
    # Verify loaded model works
    print("\n10. Verifying loaded model...")
    loaded_prediction = new_predictor.predict_individual(sample_features)
    
    # Compare predictions
    original_depression_prob = prediction_result['depression']['probability']
    loaded_depression_prob = loaded_prediction['depression']['probability']
    
    if abs(original_depression_prob - loaded_depression_prob) < 0.001:
        print("✅ Loaded model produces identical predictions")
    else:
        print("❌ Loaded model predictions differ from original")
    
    print("\n=== Demo Complete ===")
    print("The API uses the same workflow demonstrated above:")
    print("1. Initialize components on startup")
    print("2. Train models via /train endpoint")
    print("3. Make predictions via /predict endpoint")
    print("4. Manage models via model manager")
    print("\nThe model is now ready for API calls!")


def main():
    """Main demonstration function."""
    try:
        demonstrate_combined_functionality()
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()