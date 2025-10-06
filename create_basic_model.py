"""
Create a basic trained model using the custom training data for persistence.
This will be the fallback model for the Docker container.
"""

import os
import sys
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

from mental_wellness_model import MentalWellnessPredictor, DataProcessor, FeatureEngineer


def create_basic_model_from_custom_data():
    """Create and save a basic model using custom training data."""
    print("=== Creating Basic Model from Custom Training Data ===\n")
    
    # Initialize components
    print("1. Initializing components...")
    data_processor = DataProcessor()
    feature_engineer = FeatureEngineer()
    predictor = MentalWellnessPredictor(model_type='random_forest')
    
    # Load custom training data
    print("2. Loading custom training data...")
    custom_data_path = "./data/custom_training_data.csv"
    
    if not os.path.exists(custom_data_path):
        print(f"❌ Custom training data not found: {custom_data_path}")
        print("   Falling back to synthetic data...")
        df = data_processor._generate_sample_data(1000)
    else:
        try:
            df = data_processor.load_data(data_path=custom_data_path)
            print(f"   ✅ Loaded {len(df)} samples from custom data")
            print(f"   Columns: {list(df.columns)}")
        except Exception as e:
            print(f"❌ Error loading custom data: {e}")
            print("   Falling back to synthetic data...")
            df = data_processor._generate_sample_data(1000)
    
    # Process the data
    print("3. Processing data...")
    df = data_processor.clean_data(df)
    
    if not data_processor.validate_data(df):
        print("❌ Data validation failed")
        return False
    
    print(f"   ✅ Data cleaned and validated: {len(df)} samples")
    
    # Feature engineering
    print("4. Performing feature engineering...")
    df = feature_engineer.create_features(df)
    feature_engineer.fit_scaler(df)
    df = feature_engineer.transform_features(df)
    print(f"   ✅ Features engineered: {len(df.columns)} features")
    
    # Train the model
    print("5. Training model...")
    results = predictor.train(df, test_size=0.2)
    print("   ✅ Model training completed")
    
    # Display results
    print("\n📊 Training Results:")
    for condition, metrics in results.items():
        print(f"\n{condition.upper()} Model:")
        if condition in ['depression', 'anxiety']:
            print(f"  Test Accuracy: {metrics.get('test_accuracy', 0):.3f}")
            print(f"  AUC Score: {metrics.get('auc_score', 0):.3f}")
        else:  # onset_day
            print(f"  R² Score: {metrics.get('test_r2_score', 0):.3f}")
            print(f"  RMSE: {metrics.get('rmse', 0):.2f} days")
    
    # Save the model
    print("\n6. Saving basic trained model...")
    model_path = "./models/basic_trained_model.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    predictor.save_model(model_path)
    print(f"   ✅ Model saved: {model_path}")
    
    # Test the saved model
    print("7. Testing saved model...")
    test_predictor = MentalWellnessPredictor()
    test_predictor.load_model(model_path)
    
    # Test prediction with sample data
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
    
    try:
        prediction = test_predictor.predict_individual(sample_features)
        print("   ✅ Model test successful!")
        print(f"      Depression Risk: {prediction['depression']['risk_level']}")
        print(f"      Anxiety Risk: {prediction['anxiety']['risk_level']}")
        print(f"      Onset Day: {prediction.get('onset_day', {}).get('predicted_days', 'N/A')} days")
    except Exception as e:
        print(f"   ❌ Model test failed: {e}")
        return False
    
    # Get file size
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"\n📁 Model file size: {file_size:.2f} MB")
    
    print("\n=== Basic Model Creation Complete ===")
    print("✅ basic_trained_model.joblib is ready for Docker deployment")
    
    return True


def main():
    """Main function."""
    try:
        success = create_basic_model_from_custom_data()
        if success:
            print("\n🎉 Basic model created successfully!")
            print("\nThis model will be:")
            print("  - Included in Docker builds")
            print("  - Used as fallback when no GCS models available")
            print("  - Available immediately on container startup")
            print("\nYour Docker container will now have a persistent model!")
        else:
            print("\n❌ Failed to create basic model")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()