"""
Example usage of the Mental Wellness Prediction Model.
Demonstrates training and prediction capabilities.
"""

import logging
import sys
from pathlib import Path

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

from mental_wellness_model import MentalWellnessPredictor, DataProcessor, FeatureEngineer

def main():
    """Demonstrate the mental wellness prediction model."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("="*60)
    print("MENTAL WELLNESS PREDICTION MODEL - DEMO")
    print("="*60)
    
    # 1. Data Processing
    print("\n1. Loading and processing data...")
    processor = DataProcessor()
    df = processor.load_data()  # Generate sample data
    
    print(f"   Generated {len(df)} samples")
    print(f"   Features: {list(df.columns)}")
    
    # Clean and validate data
    df = processor.clean_data(df)
    if processor.validate_data(df):
        print("   Data validation: PASSED")
    
    # 2. Feature Engineering
    print("\n2. Engineering features...")
    engineer = FeatureEngineer()
    df = engineer.create_features(df)
    engineer.fit_scaler(df)
    df = engineer.transform_features(df)
    
    feature_names = engineer.get_feature_importance_names()
    print(f"   Created {len(feature_names)} features")
    
    # 3. Model Training
    print("\n3. Training prediction models...")
    predictor = MentalWellnessPredictor(model_type='random_forest')
    
    results = predictor.train(df, test_size=0.2)
    
    print("\n   TRAINING RESULTS:")
    for condition, metrics in results.items():
        print(f"   {condition.upper()}:")
        print(f"     Test Accuracy: {metrics['test_accuracy']:.3f}")
        print(f"     AUC Score: {metrics['auc_score']:.3f}")
        print(f"     CV Accuracy: {metrics['cv_mean_accuracy']:.3f} (+/- {metrics['cv_std_accuracy']:.3f})")
    
    # 4. Individual Predictions
    print("\n4. Making individual predictions...")
    
    test_cases = [
        {
            'name': 'Low Risk Individual',
            'features': {
                'age': 25,
                'sleep_hours': 8,
                'exercise_minutes': 150,
                'avg_heart_rate': 8,
                'work_stress_level': 3,
                'mood_score': 8,
                'fitness_level': 8,
                'resting_heart_rate': 60
            }
        },
        {
            'name': 'High Risk Individual',
            'features': {
                'age': 45,
                'sleep_hours': 4,
                'exercise_minutes': 30,
                'avg_heart_rate': 3,
                'work_stress_level': 9,
                'mood_score': 3,
                'fitness_level': 2,
                'resting_heart_rate': 80
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n   {test_case['name']}:")
        results = predictor.predict_individual(test_case['features'])
        
        for condition, result in results.items():
            print(f"     {condition.upper()}:")
            print(f"       Risk Level: {result['risk_level']}")
            print(f"       Probability: {result['probability']:.3f}")
    
    # 5. Feature Importance
    print("\n5. Feature importance analysis...")
    for condition in ['depression', 'anxiety']:
        importance_df = predictor.get_feature_importance(condition)
        if not importance_df.empty:
            print(f"\n   Top 5 features for {condition.upper()}:")
            for _, row in importance_df.head(5).iterrows():
                print(f"     {row['feature']}: {row['importance']:.3f}")
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Save the trained model
    model_path = "trained_mental_wellness_model.joblib"
    predictor.save_model(model_path)
    print(f"\nTrained model saved to: {model_path}")


if __name__ == '__main__':
    main()